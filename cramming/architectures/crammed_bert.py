"""This rewrite is a simplified version of the proposed changes that actually compiles statically in torch 2.0.

This model is the final, optimized crammed model.

Not all ablations discussed in the paper are implemented as switches in this version,
for all those, check scriptable_bert.py on the old branch.

"""
import copy
import torch
import random
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForTokenClassification

from typing import Optional
from omegaconf import OmegaConf

from .components import (
    _get_norm_fn,
    _get_nonlin_fn,
    EmbeddingComponent,
    PoolingComponent,
    PredictionHeadComponent,
    GLU,
    get_extended_attention_mask,
    _init_module,
)
from .attention import get_attention_mechanism


############################################################################################
# Distillation versions
############################################################################################

class distillCrammedBertConfig(PretrainedConfig):
    model_type = "distilCrammedBERT"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        self.arch = cfg_arch_container
        super().__init__(**kwargs)


def distill_construct_crammed_bert(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    config = distillCrammedBertConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    config.arch["embedding"]["vocab_size"] = vocab_size
    config.arch["num_labels"] = downstream_classes

    if downstream_classes is None:
        if config.arch["objective_layout"] == "MLM":
            model = DistillScriptableLMForPreTraining(config)
        else:
            raise ValueError(f"Invalid layout {config.arch['objective_layout']} of training objective given.")
    else:
        model = DistillScriptableLMForSequenceClassification(config)
    return model

class DistillScriptableLM(PreTrainedModel):
    """Simplified transformer wrapper. (With Distillation)"""

    config_class = distillCrammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.embedding = EmbeddingComponent(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)
        self.layers = torch.nn.ModuleList([TransformerLayer(idx, self.cfg) for idx in range(self.cfg.num_transformer_layers)])
        self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False
        self.use_causal_attention = self.cfg.attention.causal_attention

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(self.cfg.norm)(self.cfg.hidden_size, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()
        
        # Distillation point
        self.num_teacher_layers = self.cfg.num_transformer_layers
        self.student_layer_size = self.cfg.student_layer_size # divide the teacher layers by this number
        self.distill_point = self.num_teacher_layers // self.student_layer_size
        self.random_distill = self.cfg.random_distill

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None,
                 compute_distillation: bool = False, double_pass: bool = False):
        
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.use_causal_attention)
        hidden_states = self.embedding(input_ids)

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        intermediate_output = None
        distill_point = None

        def process_layers(hidden_states, distill_point_arg):
            nonlocal intermediate_output
            for i, layer_module in enumerate(self.layers):
                hidden_states = layer_module(hidden_states, attention_mask)
                if i + 1 == distill_point_arg:
                    intermediate_output = hidden_states.clone()
            return hidden_states
        
        # Pick the distillation point
        if self.random_distill:
            distill_point = random.randint(1, self.num_teacher_layers) # Random distillation point to distill to from the teacher
        else:
            distill_point = self.distill_point # Fixed distillation point (default)

        # First pass
        hidden_states = process_layers(hidden_states, distill_point)
        
        # Second pass if double_pass is True
        if double_pass:
            hidden_states = process_layers(hidden_states, distill_point)
        
        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            if intermediate_output is not None:
                intermediate_output = intermediate_output.transpose(0, 1).contiguous()
        
        intermediate_output = self.final_norm(intermediate_output)
        final_output = self.final_norm(hidden_states)

        return final_output, intermediate_output
    
    def get_student_model(self):
        student_cfg = copy.deepcopy(self.config)
        student_cfg.arch['num_transformer_layers'] = self.cfg.load_student_layers
        student_model = DistillScriptableLM(student_cfg)
        student_model.embedding = self.embedding
        student_model.layers = torch.nn.ModuleList(self.layers[:self.cfg.load_student_layers])
        student_model.final_norm = self.final_norm
        return student_model


class DistillScriptableLMForPreTraining(PreTrainedModel):
    """Pretraining version with optional prediction head and variant for sparse prediction. (With Distillation)"""

    config_class = distillCrammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = DistillScriptableLM(config)

        if not self.cfg.skip_head_transform:
            self.prediction_head = PredictionHeadComponent(self.cfg)
        else:
            self.prediction_head = torch.nn.Identity()  # from linear in old version

        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.mlm_loss_fn = torch.nn.CrossEntropyLoss() #ignore index -100?
        self.sparse_prediction = self.cfg.sparse_prediction

        # Distillation Loss

        if self.cfg.distill_type == "cutoff":
            self.distillation_loss_fn = self.compute_distillbert_loss
        elif self.cfg.distill_type == "skpd":
            self.distillation_loss_fn = self.compute_skpd_distillation_loss
        else:
            self.distillation_loss_fn = self.compute_distillation_loss

        self.distillation_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.cos_loss = torch.nn.CosineEmbeddingLoss(reduction='mean')
        self.temperature = self.cfg.temperature # Temperature
        self.alpha_ce = self.cfg.alpha_ce  # Weight for soft distillation loss
        self.alpha_mlm = self.cfg.alpha_mlm  # Weight for MLM loss
        self.alpha_cos = self.cfg.alpha_cos  # Weight for cosine embedding loss
        self.temperature_squared = self.temperature ** 2
        self._init_weights()

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, compute_distillation: bool = False):
        
        final_outputs, intermediate_outputs = self.encoder(input_ids, attention_mask)
        
        final_outputs = self.prediction_head(final_outputs)
        final_logits = self.decoder(final_outputs)

        intermediate_outputs = self.prediction_head(intermediate_outputs)
        intermediate_logits = self.decoder(intermediate_outputs)
        
        loss_dict = dict()
        if labels is not None:
            teacher_mlm_loss = self.mlm_loss_fn(final_logits.view(-1, final_logits.size(-1)), labels.view(-1))
            student_mlm_loss = self.mlm_loss_fn(intermediate_logits.view(-1, intermediate_logits.size(-1)), labels.view(-1))
            loss_dict["teacher_mlm_loss"] = teacher_mlm_loss
            loss_dict["student_mlm_loss"] = student_mlm_loss

            if compute_distillation and intermediate_outputs is not None:
                distill_loss_dict = self.distillation_loss_fn(final_logits, intermediate_logits, labels, final_outputs, intermediate_outputs, student_mlm_loss)
                loss_dict.update(distill_loss_dict)

        return {
            "teacher_mlm_loss": loss_dict.get("teacher_mlm_loss"),
            "logits": final_logits,
            "intermediate_logits": intermediate_logits,
            "student_mlm_loss": loss_dict.get("student_mlm_loss", torch.tensor(0.0, device=final_logits.device)),
            "distillation_loss": loss_dict.get("distillation_loss", torch.tensor(0.0, device=final_logits.device)),
        }

    def compute_distillbert_loss(self, teacher_logits, student_logits, labels, teacher_hidden_states, student_hidden_states, student_mlm_loss):

        # Soft distillation loss
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature_squared)

        # Cosine embedding loss
        batch_size, seq_length, hidden_dim = student_hidden_states.size()
        cos_loss = self.cos_loss(
            student_hidden_states.view(-1, hidden_dim),
            teacher_hidden_states.view(-1, hidden_dim),
            torch.ones(batch_size * seq_length).to(student_hidden_states.device)
        )

        # Combine losses
        distillation_loss = (
            self.alpha_ce * soft_loss +
            self.alpha_mlm * student_mlm_loss +
            self.alpha_cos * cos_loss
        )
        
        return {
            "distillation_loss": distillation_loss,
        }
    
    def compute_distillation_loss(self, teacher_logits, student_logits, labels, teacher_hidden_states, student_hidden_states, student_mlm_loss):

        # Soft distillation loss
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
    
        # Combine losses
        distillation_loss = (
            self.alpha_ce * soft_loss +
            self.alpha_mlm * student_mlm_loss
        )
        
        return {
            "distillation_loss": distillation_loss,
        }
    
    def compute_skpd_distillation_loss(self, teacher_logits, student_logits, labels, teacher_hidden_states, student_hidden_states, student_mlm_loss):
        
        # Soft distillation loss
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature_squared)
    
        # SKPD loss
        teacher_hidden_states = teacher_hidden_states.view(teacher_hidden_states.size(0), -1)
        student_hidden_states = student_hidden_states.view(student_hidden_states.size(0), -1)

        teacher_hidden_states = F.normalize(teacher_hidden_states, p=2, dim=-1)
        student_hidden_states = F.normalize(student_hidden_states, p=2, dim=-1)

        teacher_similarities = torch.mm(teacher_hidden_states, teacher_hidden_states.t())
        student_similarities = torch.mm(student_hidden_states, student_hidden_states.t())

        skpd_loss = F.mse_loss(teacher_similarities, student_similarities)

        # Combine losses
        distillation_loss = (
            self.alpha_ce * soft_loss +
            self.alpha_mlm * student_mlm_loss +
            self.alpha_cos * skpd_loss
        )
        
        return {
            "distillation_loss": distillation_loss,
        }
    
    def freeze_layers(self, layers_to_freeze: int):
        for i, layer in enumerate(self.encoder.layers):
            if i < layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def unfreeze_layers(self, layers_to_unfreeze: int):
        for i, layer in enumerate(self.encoder.layers):
            if i < layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True


    def _forward_sparse(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):
        labels = labels.view(-1)
        mask_positions = labels != self.mlm_loss_fn.ignore_index
        num_masks_guaranteed = round(self.sparse_prediction * labels.shape[0])
        indices = torch.argsort(mask_positions.int())[-num_masks_guaranteed:]

        outputs = outputs.view(-1, outputs.size(-1))[indices]
        labels = labels[indices]

        masked_lm_loss = self.mlm_loss_fn(outputs, labels)
        return masked_lm_loss

class DistillScriptableLMForSequenceClassification(PreTrainedModel):
    """Classification head and pooler."""

    config_class = distillCrammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.num_labels = self.cfg.num_labels
        self.student_output = self.cfg.student_output

        self.encoder = DistillScriptableLM(config)
        self.pooler = PoolingComponent(self.cfg.classification_head, self.cfg.hidden_size)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

        self.problem_type = None
        self._init_weights()

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        
        encoder_output = self.encoder(input_ids, attention_mask)
        if isinstance(encoder_output, tuple):
            if self.student_output:
                hidden_states = encoder_output[-1] # student output
            else:
                hidden_states = encoder_output[0] # teacher output
        else:
            hidden_states = encoder_output

        final_logits = self.head(self.pooler(hidden_states))

        if labels is not None:
            if self.problem_type is None:  # very much from huggingface
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(final_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(final_logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(final_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(final_logits, labels)
        else:
            loss = final_logits.new_zeros((1,))

        return dict(logits=final_logits, loss=loss)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        load_full_model = kwargs.get('load_full_model', False)
        config = kwargs.get('config', None)
        
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        model = cls(config)
        
        # Load the state dict
        state_dict = torch.load(pretrained_model_name_or_path)
        
        if load_full_model == False:
            # Get the student model
            student_encoder = model.encoder.get_student_model()
            model.encoder = student_encoder
        
        model.load_state_dict(state_dict, strict=False)
        return model


############################################################################################
# Cramming Versions
############################################################################################

class crammedBertConfig(PretrainedConfig):
    model_type = "crammedBERT"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        self.arch = cfg_arch_container
        super().__init__(**kwargs)



def construct_crammed_bert(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    config = crammedBertConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    config.arch["embedding"]["vocab_size"] = vocab_size
    config.arch["num_labels"] = downstream_classes

    if downstream_classes is None:
        if config.arch["objective_layout"] == "MLM":
            model = ScriptableLMForPreTraining(config)
        elif config.arch["objective_layout"] == "SCRIPT":
            model = ScriptableLMForSCRIPTTraining(config)
        else:
            raise ValueError(f"Invalid layout {config.arch['objective_layout']} of training objective given.")
    else:
        model = ScriptableLMForSequenceClassification(config)
    return model


class AttentionComponent(torch.nn.Module):
    def __init__(self, idx, hidden_size, cfg_attention, use_bias=True):
        super().__init__()
        self.self_attention = get_attention_mechanism(idx, hidden_size, cfg_attention)
        if cfg_attention.skip_output_projection:
            self.dense = torch.nn.Identity()
        else:
            self.dense = torch.nn.Linear(self.self_attention.output_dim, hidden_size, bias=use_bias)

        self.LAYOUT = self.self_attention.LAYOUT

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return self.dense(self.self_attention(hidden_states, attention_mask))


class FFNComponent(torch.nn.Module):
    """Note: The FF layer is not auto-scaled when using a GLU type activation.
    It actually turned out better not to scale it, so here the block is effectively smaller than may be expected.

    The neox suggestion for approx. equal parameter count is int(4 * 2 / 3 * hidden_size) * 2 [this is ~5.33]
    """

    def __init__(self, hidden_size, intermed_size, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()
        self.dense_in = torch.nn.Linear(hidden_size, intermed_size, bias=use_bias)
        self.nonlin = nonlin_fn()
        if isinstance(self.nonlin, GLU):
            intermed_output_size = intermed_size // 2
        else:
            intermed_output_size = intermed_size
        self.dense_out = torch.nn.Linear(intermed_output_size, hidden_size, bias=use_bias)

    def forward(self, hidden_states):
        return self.dense_out(self.nonlin(self.dense_in(hidden_states)))


class TransformerLayer(torch.nn.Module):
    """A transformer-encoder structure based on the components from above."""

    def __init__(self, idx, cfg_arch):
        super().__init__()
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=False)
        self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.attn = AttentionComponent(
            idx,
            cfg_arch.hidden_size,
            cfg_arch.attention,
            cfg_arch.use_bias,
        )
        self.LAYOUT = self.attn.LAYOUT

        self.ffn = FFNComponent(
            cfg_arch.hidden_size,
            cfg_arch.intermed_size,
            _get_nonlin_fn(cfg_arch.nonlin),
            cfg_arch.use_bias,
        )

    def forward(self, states, attention_mask: Optional[torch.Tensor] = None):
        states = states + self.dropout(self.attn(self.norm1(states), attention_mask))
        states = states + self.dropout(self.ffn(self.norm2(states)))
        return states


class ScriptableLM(PreTrainedModel):
    """Simplified transformer wrapper."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.embedding = EmbeddingComponent(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)
        self.layers = torch.nn.ModuleList([TransformerLayer(idx, self.cfg) for idx in range(self.cfg.num_transformer_layers)])
        self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False
        self.use_causal_attention = self.cfg.attention.causal_attention

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(self.cfg.norm)(self.cfg.hidden_size, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()
        
        #self.distill_point = self.cfg.num_transformer_layers // self.cfg.student_layer_size

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.use_causal_attention)
        hidden_states = self.embedding(input_ids)

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, attention_mask)

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        return self.final_norm(hidden_states)
    
    def get_student_model(self):
        student_cfg = copy.deepcopy(self.config)
        num_student_layers = self.cfg.load_student_layers
        student_cfg.arch['num_transformer_layers'] = num_student_layers
        student_model = ScriptableLM(student_cfg)
        student_model.embedding = self.embedding
        student_model.layers = torch.nn.ModuleList(self.layers[:num_student_layers])
        student_model.final_norm = self.final_norm
        return student_model


class ScriptableLMForPreTraining(PreTrainedModel):
    """Pretraining version with optional prediction head and variant for sparse prediction."""

    config_class = distillCrammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableLM(config)

        if not self.cfg.skip_head_transform:
            self.prediction_head = PredictionHeadComponent(self.cfg)
        else:
            self.prediction_head = torch.nn.Identity()  # from linear in old version

        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sparse_prediction = self.cfg.sparse_prediction

        self._init_weights()

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        outputs = self.encoder(input_ids, attention_mask)
        outputs = outputs.view(-1, outputs.shape[-1])

        if self.sparse_prediction and labels is not None:
            masked_lm_loss = self._forward_sparse(outputs, labels)
        else:
            outputs = self.decoder(self.prediction_head(outputs))
            if labels is not None:
                masked_lm_loss = self.loss_fn(outputs, labels.view(-1))
            else:
                masked_lm_loss = outputs.new_zeros((1,))

        return {"loss": masked_lm_loss, "outputs": outputs}

    # Sparse prediction usually has an unpredictable number of entries in each batch
    # but the dataloader was modified so that 25% of the batch is ALWAYS masked.
    # This allows for static compilation. If you modify the dataloader, this function will fill your compile cache
    def _forward_sparse(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):

        labels = labels.view(-1)
        mask_positions = labels.view(-1) != self.loss_fn.ignore_index
        num_masks_guaranteed = round(self.sparse_prediction * labels.shape[0])
        # outputs = outputs[mask_positions]  # not allowed as dynamic shape op
        # labels = labels[mask_positions]
        # torch.masked_select(labels, mask_positions)  # not allowed as a dynamic shape operator

        # indices = torch.arange(mask_positions.shape[0], device=outputs.device)[mask_positions] # not allowed
        indices = torch.argsort(mask_positions.int())[-num_masks_guaranteed:]  # ugh

        outputs = outputs[indices]  # not allowed as dynamic shape op, but ok with indices
        labels = labels[indices]
        # alternative:
        # outputs = torch.take_along_dim(outputs, indices.view(-1, 1), 0)
        # labels = torch.take(labels, indices)

        outputs = self.decoder(self.prediction_head(outputs))
        masked_lm_loss = self.loss_fn(outputs, labels)
        return masked_lm_loss


class ScriptableLMForSequenceClassification(PreTrainedModel):
    """Classification head and pooler."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.num_labels = self.cfg.num_labels

        self.encoder = ScriptableLM(config)
        self.pooler = PoolingComponent(self.cfg.classification_head, self.cfg.hidden_size)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

        self.problem_type = None
        self._init_weights()

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        logits = self.head(self.pooler(self.encoder(input_ids, attention_mask)))

        if labels is not None:
            if self.problem_type is None:  # very much from huggingface
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        else:
            loss = logits.new_zeros((1,))

        return dict(logits=logits, loss=loss)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        load_full_model = kwargs.get('load_full_model', False)
        config = kwargs.get('config', None)
        
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        model = cls(config)
        
        # Load the state dict
        state_dict = torch.load(pretrained_model_name_or_path)
        
        if load_full_model == False:
            # Get the student model
            student_encoder = model.encoder.get_student_model()
            model.encoder = student_encoder
        
        model.load_state_dict(state_dict, strict=False)
        return model


class ScriptableLMForSCRIPTTraining(PreTrainedModel):
    """Pretraining machinery using SCRIPT from Nijkamp et al., 2021. Always running sparse prediction."""

    config_class = crammedBertConfig
    ALPHA = 1.0  # SCRIPT constant

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.num_labels = self.cfg.num_labels

        self.encoder = ScriptableLM(config)
        self.prediction_head = PredictionHeadComponent(self.cfg)

        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sparse_prediction = self.cfg.sparse_prediction
        assert self.sparse_prediction

        self._init_weights()

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        loss = torch.tensor(0.0, dtype=torch.float, device=input_ids.device)

        outputs = self.encoder(input_ids, attention_mask)
        outputs = outputs.view(-1, outputs.shape[-1])

        if labels is not None:
            # ## Generation pass ##
            labels = labels.view(-1)
            mask_positions = labels.view(-1) != self.loss_fn.ignore_index
            num_masks_guaranteed = round(self.sparse_prediction * labels.shape[0])
            indices = torch.argsort(mask_positions.int())[-num_masks_guaranteed:]

            # sparse outputs for prediction
            outputs = outputs[indices]
            labels = labels[indices]

            logits = self.decoder(self.prediction_head(outputs))  # sparse logits
            loss += self.loss_fn(logits, labels)

            # ## Discrimination pass ##
            resampled_token_ids = self._gumbel_sample(logits.detach())
            discriminator_input_ids = input_ids.clone().view(-1)
            discriminator_input_ids[indices] = resampled_token_ids

            critic_labels = (input_ids.view(-1) != discriminator_input_ids).to(outputs.dtype)

            outputs = self.encoder(discriminator_input_ids.view_as(input_ids), attention_mask).view(-1, outputs.shape[-1])
            disc_logits = self.decoder(self.prediction_head(outputs))  # full logits
            binary_logits = self._get_binary_logits(disc_logits)

            # ELECTRA-type discriminator:
            loss += self.ALPHA * torch.nn.functional.binary_cross_entropy_with_logits(binary_logits, critic_labels)

        else:
            logits = self.decoder(self.prediction_head(outputs))
            loss += outputs.new_zeros((1,))

        return {"loss": loss, "logits": logits}

    def _get_binary_logits(self, logits):
        # Convert to binary decision as described in SCRIPT
        # exp_logitsum = torch.exp(disc_logits).sum(dim=-1)  # autocast ok?
        # binary_logits = torch.stack([1 / (exp_logitsum + 1), exp_logitsum / (exp_logitsum + 1)], dim=-1)  # stack minus and plus
        # instead, we can also compute logit[binary_logits], which is

        # let y = sum(exp(logits)) / ( sum(exp(logits))+1 ), 1-y = 1 / ( sum(exp(logits))+1 )
        # log(y / (1-y)) = log( sum(exp(logits)) / ( sum(exp(logits))+1 ) * ( sum(exp(logits))+1 ) / 1)
        #                = log(sum(exp(logits))
        # Then, we can use BCEWithLogitsLoss, to safely compute logit probs via sigmoids
        return torch.logsumexp(logits, dim=-1)

    def _gumbel_sample(self, logits, temperature=1.0):
        """via https://github.com/lucidrains/electra-pytorch/blob/master/electra_pytorch/electra_pytorch.py"""
        return ((logits / temperature) + self._gumbel_noise(logits)).argmax(dim=-1)

    def _gumbel_noise(self, inputs, eps=1e-9):
        """via https://github.com/lucidrains/electra-pytorch/blob/master/electra_pytorch/electra_pytorch.py"""
        noise = torch.zeros_like(inputs).uniform_(0, 1)
        return -torch.log(-torch.log(noise + eps) + eps)


class ScriptableLMForTokenClassification(PreTrainedModel):
    """Classification head without pooling."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableLM(config)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

        self.problem_type = None
        self._init_weights()

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        logits = self.head(self.encoder(input_ids, attention_mask))

        if labels is not None:
            if self.problem_type is None:  # very much from huggingface
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            else:
                raise ValueError("Wrong problem type!")
        else:
            loss = logits.new_zeros((1,))

        return dict(logits=logits, loss=loss)


# ###### HF registry here ############### #

#AutoConfig.register("crammedBERT", crammedBertConfig)
#AutoModel.register(crammedBertConfig, ScriptableLM)
#AutoModelForMaskedLM.register(crammedBertConfig, ScriptableLMForPreTraining)
#AutoModelForSequenceClassification.register(crammedBertConfig, ScriptableLMForSequenceClassification)
#AutoModelForTokenClassification.register(crammedBertConfig, ScriptableLMForTokenClassification)

AutoConfig.register("distilCrammedBERT", distillCrammedBertConfig)
AutoModel.register(distillCrammedBertConfig, DistillScriptableLM)
AutoModelForMaskedLM.register(distillCrammedBertConfig, DistillScriptableLMForPreTraining)
