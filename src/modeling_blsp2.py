import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

import logging
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import WhisperConfig


from .plora import LoraConfig, LoraModel
from .modeling_adapter import Subsampler, CFormer
from .configuration_blsp2 import Blsp2Config
from .configuration_qwen import QWenConfig
from .modeling_utils import length_to_attention_mask, check_shape
from .modeling_whisper_encoder import WhisperEncoder
from .modeling_qwen import QWenLMHeadModel
import torch.nn.functional as F


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

text_llm_related_losses = {"response_kl", "input_kl"}
speech_llm_related_losses = {"response_kl", "input_kl", "response_ce", "input_er"}
lm_related_losses = text_llm_related_losses | speech_llm_related_losses


class Blsp2Model(PreTrainedModel):
    config_class = Blsp2Config
    base_model_prefix = "blsp2"

    def __init__(self, config: Blsp2Config):
        super().__init__(config)
        self.whisper_config = WhisperConfig(**config.whisper_config)
        self.qwen_config = QWenConfig(**config.qwen_config)

        self.whisper_model = WhisperEncoder(self.whisper_config)
        self.qwen_model = QWenLMHeadModel(self.qwen_config)

        if config.lora_config:
            self.lora_config = LoraConfig(**config.lora_config)
            self.qwen_model = LoraModel(self.qwen_model, self.lora_config, "default")

        if config.adapter_type == "subsampler":
            self.adapter = Subsampler(self.whisper_config.d_model, config.adapter_inner_dim, self.qwen_config.hidden_size,
                                      config.adapter_hidden_layers, self.whisper_config, config.conv_kernel_sizes)

        elif config.adapter_type == "cformer":
            self.adapter = CFormer(self.whisper_config, self.qwen_config.hidden_size,
                                   self.qwen_config.vocab_size,
                                   num_pre_cif_layers=config.num_pre_cif_layers,
                                   num_post_cif_layers=config.num_post_cif_layers)
        else:
            raise ValueError(f"unsupported adapter type: {config.adapter_type}")
        
        self.hidden2emotion = nn.Linear(self.qwen_config.hidden_size, self.config.num_emotions, bias=False)

        self.loss_names = [] # must be a list of loss names:  seq_kd, token_kd, or others before training

    def set_loss_names(self, names):
        self.loss_names = names

    def forward(
        self,
        start_ids: torch.LongTensor,
        start_mask: torch.Tensor,
        start_labels: torch.LongTensor,
        instruction_ids: torch.LongTensor,
        instruction_mask: torch.Tensor,
        instruction_labels: torch.LongTensor,
        audio_instruction_ids: torch.LongTensor,
        audio_instruction_mask: torch.Tensor,
        audio_instruction_labels: torch.LongTensor,
        input_ids: torch.LongTensor,
        input_mask: torch.Tensor,
        input_labels: torch.LongTensor,
        speech_values: torch.FloatTensor,
        speech_mask: torch.LongTensor,
        suffix_ids: torch.LongTensor,
        suffix_mask: torch.Tensor,
        suffix_labels: torch.LongTensor,
        emotion_labels: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        assert len(self.loss_names) > 0, "self.loss_names cannot be empty"

        if not any ("response" in loss_name for loss_name in self.loss_names):
            batch_size = start_ids.size(0)
            instruction_ids = torch.zeros(batch_size, 0, dtype=start_ids.dtype, device=start_ids.device)
            instruction_mask = torch.zeros(batch_size, 0, dtype=start_mask.dtype, device=start_mask.device)
            instruction_labels = torch.zeros(batch_size, 0, dtype=start_labels.dtype, device=start_labels.device)
            audio_instruction_ids = torch.zeros(batch_size, 0, dtype=start_ids.dtype, device=start_ids.device)
            audio_instruction_mask = torch.zeros(batch_size, 0, dtype=start_mask.dtype, device=start_mask.device)
            audio_instruction_labels = torch.zeros(batch_size, 0, dtype=start_labels.dtype, device=start_labels.device)
            suffix_ids = torch.zeros(batch_size, 0, dtype=start_ids.dtype, device=start_ids.device)
            suffix_mask = torch.zeros(batch_size, 0, dtype=start_mask.dtype, device=start_mask.device)
            suffix_labels = torch.zeros(batch_size, 0, dtype=start_labels.dtype, device=start_labels.device)

        
        start_embeds = self.qwen_model.get_input_embeddings()(start_ids)
        instruction_embeds = self.qwen_model.get_input_embeddings()(instruction_ids)
        audio_instruction_embeds = self.qwen_model.get_input_embeddings()(audio_instruction_ids)
        input_embeds = self.qwen_model.get_input_embeddings()(input_ids)
        suffix_embeds = self.qwen_model.get_input_embeddings()(suffix_ids)

        speech_input_embeds, speech_input_mask, speech_input_logits, speech_cif_alphas, speech_pred_num_tokens = \
            self.get_speech_features(speech_values, speech_mask, input_mask.sum(-1))
        speech_input_labels = speech_input_mask.new_ones(speech_input_embeds.size(0), speech_input_embeds.size(1),
                                                         dtype=torch.int64).fill_(-100)

        speech_embeds = torch.cat([start_embeds, audio_instruction_embeds, speech_input_embeds, suffix_embeds], dim=1)
        speech_mask = torch.cat([start_mask, audio_instruction_mask, speech_input_mask, suffix_mask], dim=1)
        speech_labels = torch.cat([start_labels, audio_instruction_labels, speech_input_labels, suffix_labels], dim=1)

        if any(loss_name in text_llm_related_losses for loss_name in self.loss_names):
            text_embeds = torch.cat([start_embeds, instruction_embeds, input_embeds, suffix_embeds], dim=1)
            text_mask = torch.cat([start_mask, instruction_mask, input_mask, suffix_mask], dim=1)
            text_labels = torch.cat([start_labels, instruction_labels, input_labels, suffix_labels], dim=1)
            input_kd_labels = torch.cat([torch.zeros_like(start_labels),
                                         torch.zeros_like(instruction_labels),
                                         input_mask,
                                         torch.zeros_like(suffix_labels)], dim=1)
            speech_kd_labels = torch.cat([torch.zeros_like(start_labels),
                                          torch.zeros_like(audio_instruction_labels),
                                          input_mask,
                                          torch.zeros_like(suffix_labels)], dim=1)
            text_response_kd_labels = torch.cat([torch.zeros_like(start_labels),
                                                 torch.zeros_like(instruction_labels),
                                                 torch.zeros_like(input_labels),
                                                 (suffix_labels != -100).long()], dim=1)
            speech_response_kd_labels = torch.cat([torch.zeros_like(start_labels),
                                                   torch.zeros_like(audio_instruction_labels),
                                                   torch.zeros_like(speech_input_labels),
                                                   (suffix_labels != -100).long()], dim=1)
            lora_audio_mask = torch.zeros_like(text_labels)
            self.update_lora_mask(lora_audio_mask, False)
            with torch.no_grad():
                text_output = self.qwen_model(inputs_embeds=text_embeds, attention_mask=text_mask,
                                              position_ids=text_mask.cumsum(dim=-1) - 1, output_hidden_states=True,
                                              return_dict=True)
                text_logits = text_output.logits
        if any(loss_name in speech_llm_related_losses for loss_name in self.loss_names):
            lora_audio_mask = torch.cat([torch.zeros_like(start_mask),
                                         torch.zeros_like(audio_instruction_mask),
                                         torch.ones_like(speech_input_mask),
                                         torch.zeros_like(suffix_mask)], dim=1)
            self.update_lora_mask(lora_audio_mask, False)
            speech_output = self.qwen_model(inputs_embeds=speech_embeds, attention_mask=speech_mask,
                                            position_ids=speech_mask.cumsum(dim=-1) - 1, output_hidden_states=True,
                                            return_dict=True)
            speech_logits = speech_output.logits

        total_loss = input_embeds.new_zeros(())
        for loss_name in self.loss_names:
            if loss_name == "response_ce":
                shifted_logits = speech_logits[..., :-1, :].contiguous()
                shifted_labels = speech_labels[..., 1:].contiguous()
                loss = F.cross_entropy(shifted_logits[shifted_labels != -100],
                                       shifted_labels[shifted_labels != -100], reduction="mean")
                total_loss += loss
            elif loss_name == "response_kl":
                loss = F.kl_div(
                    F.log_softmax(speech_logits[speech_response_kd_labels == 1] / self.config.kd_temperature, dim=-1),
                    F.softmax(text_logits[text_response_kd_labels == 1] / self.config.kd_temperature, dim=-1),
                    reduction="batchmean"
                )
                total_loss += loss
            elif loss_name == "input_kl":
                check_shape(input_labels, speech_input_labels)
                loss = F.kl_div(
                    F.log_softmax(speech_logits[speech_kd_labels == 1] / self.config.kd_temperature, dim=-1),
                    F.softmax(text_logits[input_kd_labels == 1] / self.config.kd_temperature, dim=-1),
                    reduction="batchmean"
                )
                total_loss += loss
            elif loss_name == "cif":
                if speech_pred_num_tokens is None:
                    raise RuntimeError("predicted_num_tokens not set but cif_loss is requested")
                loss = F.l1_loss(speech_pred_num_tokens/input_mask.sum(-1), torch.ones_like(speech_pred_num_tokens),
                                  reduction="mean")
                total_loss += loss
                # loss_str += f"{loss_name}: {loss.item():.4f}, "
            elif loss_name == "input_er":
                hidden_states = speech_input_embeds.clone()
                hidden_states[speech_input_mask == 0] = 0.0
                pooled_output = hidden_states.sum(dim=1) / speech_input_mask.sum(dim=1).view(-1, 1)
                er_logits = self.hidden2emotion(pooled_output)
                loss = F.cross_entropy(er_logits.view(-1, self.config.num_emotions), emotion_labels.view(-1))
                total_loss += loss
            else:
                raise RuntimeError(f"Unsupported loss name: {loss_name}")

        return {"loss": total_loss}

    def add_lora(self, lora_config, lora_scope="global"):
        if self.config.lora_config:
            logger.warning(f"add_lora ignored as model already has lora enabled")
        else:
            self.lora_config = lora_config
            self.config.lora_config = lora_config.to_dict()
            self.qwen_model = LoraModel(self.qwen_model, self.lora_config, "default")
            self.config.lora_scope = lora_scope

    def update_lora_mask(self, audio_mask, inference_mode: bool):
        if not self.config.lora_config or self.config.lora_scope == "global":
            return

        self.qwen_model.update_inference_mode(inference_mode)
        if self.config.lora_scope == "audio":
            self.qwen_model.update_lora_mask("default", audio_mask)
        elif self.config.lora_scope == "text":
            self.qwen_model.update_lora_mask("default", torch.ones_like(audio_mask) - audio_mask)
        elif self.config.lora_scope == "global":
            pass # do nonthing as official peft uses global lora
        else:
            raise ValueError(f"The scope value {self.config.lora_scope} for lora adapter 'default' is not supported")

    def merge_lora(self):
        if hasattr(self, 'lora_config'):
            if self.config.lora_scope != "global":
                raise ValueError(f"cannot call merge_lora when the lora_scope is not global ("
                                 f"{self.config.lora_scope})")
            self.qwen_model = self.qwen_model.merge_and_unload()
            self.config.lora_config = {}
            del self.lora_config
        else:
            raise ValueError("cannot call merge_lora when no self.lora_config is set")

    def get_speech_features(self, speech_values, speech_attention_mask, num_tokens=None):
        w2v_args = {
            "input_features": speech_values,
            "attention_mask": speech_attention_mask,
        }
        output = self.whisper_model(**w2v_args)
        speech_embeds = output.last_hidden_state # B x T x C
        attention_mask = length_to_attention_mask(output.output_lengths)

        speech_embeds, speech_atts, speech_logits, speech_cif_alphas, speech_pred_num_tokens = \
            self.adapter(speech_embeds, attention_mask, num_tokens)

        return speech_embeds, speech_atts, speech_logits, speech_cif_alphas, speech_pred_num_tokens

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        suffix_input_ids,
        suffix_attention_mask,
        speech_values=None,
        speech_attention_mask=None,
        generation_config=None,
        stop_words_ids=None
    ):
        inputs_embeds, input_attention_mask, lora_audio_mask = [], [], []

        prefix_embeds = self.qwen_model.get_input_embeddings()(input_ids)
        inputs_embeds.append(prefix_embeds)
        input_attention_mask.append(attention_mask)
        lora_audio_mask.append(torch.zeros_like(attention_mask))

        if speech_values is not None:
            speech_embeds, speech_attention_mask, _, _, _ = self.get_speech_features(speech_values, speech_attention_mask)
            inputs_embeds.append(speech_embeds)
            input_attention_mask.append(speech_attention_mask)
            lora_audio_mask.append(torch.ones_like(speech_attention_mask))

        suffix_embeds = self.qwen_model.get_input_embeddings()(suffix_input_ids)
        inputs_embeds.append(suffix_embeds)
        input_attention_mask.append(suffix_attention_mask)
        lora_audio_mask.append(torch.zeros_like(suffix_attention_mask))

        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        input_attention_mask = torch.cat(input_attention_mask, dim=1)
        lora_audio_mask = torch.cat(lora_audio_mask, dim=1)

        self.update_lora_mask(lora_audio_mask, True)

        return self.qwen_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=input_attention_mask,
            generation_config=generation_config,
            stop_words_ids=stop_words_ids
        )
    
    @torch.no_grad()
    def chat(
        self,
        history,
        generation_config,
        stop_words_ids,
        device,
    ):
        inputs_embeds = []
        lora_audio_mask = []

        for h in history:
            if len(h) == 1:
                ### text
                input_ids = h[0].to(device)
                embeds = self.qwen_model.get_input_embeddings()(input_ids)
                inputs_embeds.append(embeds)
                lora_audio_mask.append(torch.zeros_like(input_ids))
            elif len(h) == 2:
                ### speech
                speech_values, speech_attention_mask = h[0].to(device), h[1].to(device)
                speech_embeds, speech_attention_mask, _, _, _= self.get_speech_features(speech_values, speech_attention_mask)
                inputs_embeds.append(speech_embeds)
                lora_audio_mask.append(speech_attention_mask)
            else:
                raise NotImplementedError
        
        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        lora_audio_mask = torch.cat(lora_audio_mask, dim=1)
        self.update_lora_mask(lora_audio_mask, True)

        return self.qwen_model.generate(
            inputs_embeds=inputs_embeds,
            generation_config=generation_config,
            stop_words_ids=stop_words_ids
        )
