import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

import logging
from transformers import WhisperConfig
from .modeling_utils import length_to_4d_attention_mask, length_to_attention_mask
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

logger = logging.getLogger(__name__)


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)
    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


class Subsampler(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int,
        num_hidden_layers: int,
        whisper_config,
        conv_kernel_sizes: str="5,5,5",
        num_hidden_heads: int = 8,
    ):
        super(Subsampler, self).__init__()
        self.subsampler = Conv1dSubsampler(
            in_dim,
            2 * in_dim,
            out_dim,
            [int(k) for k in conv_kernel_sizes.split(",")],
        )

        self.fc1 = nn.Linear(out_dim, mid_dim, bias=False)
        self.fc2 = nn.Linear(mid_dim, out_dim, bias=False)
        self.num_hidden_layers = num_hidden_layers
        if num_hidden_layers > 0:
            config_dict = whisper_config.to_dict()
            config_dict['d_model'] = mid_dim
            config_dict['encoder_attention_heads'] = num_hidden_heads
            self.hidden_layers = nn.ModuleList([WhisperEncoderLayer(WhisperConfig(**config_dict)) for _ in range(num_hidden_layers)])
        else:
            self.activation = nn.GELU()
        self.speech_ln = torch.nn.LayerNorm(out_dim, 1e-5, True)

    def forward(self, x, attention_mask, num_tokens=None):
        x, lengths = self.subsampler(x, attention_mask.sum(dim=-1)) # B x T x H -> T x B x C
        attention_mask = length_to_attention_mask(lengths)
        x = x.transpose(0,1) # T x B x C -> B x T x C

        residual = x
        x = self.fc1(x)
        if self.num_hidden_layers > 0:
            # x = x.transpose(0,1) # B x T x C -> T x B x C
            for layer in self.hidden_layers:
                x = layer(x, None, None, False)[0]
            # x = x.transpose(0,1) # T x B x C -> B x T x C
        else:
            x = self.activation(x)
        x = self.fc2(x) + residual

        x = self.speech_ln(x)

        return x, attention_mask, None, None, None


class CFormer(nn.Module):
    def __init__(
            self,
            whisper_config,
            out_dim: int,
            vocab_size: int,
            num_pre_cif_layers: int = 1,
            num_post_cif_layers: int = 1,
            num_hidden_heads: int = 8,
    ):
        super(CFormer, self).__init__()
        self.num_pre_cif_layers = num_pre_cif_layers
        if num_pre_cif_layers > 0:
            config_dict = whisper_config.to_dict()
            config_dict['d_model'] = whisper_config.d_model
            config_dict['encoder_attention_heads'] = num_hidden_heads
            self.pre_cif_layers = nn.ModuleList([
                WhisperEncoderLayer(WhisperConfig(**config_dict)) for _ in range(num_pre_cif_layers)
            ])
        else:
            self.pre_cif_layer = nn.Linear(whisper_config.d_model, whisper_config.d_model)
        self.cif_proj = nn.Linear(whisper_config.d_model - 1, whisper_config.d_model)
        self.num_post_cif_layers = num_post_cif_layers
        if num_post_cif_layers > 0:
            config_dict = whisper_config.to_dict()
            config_dict['d_model'] = whisper_config.d_model
            config_dict['encoder_attention_heads'] = num_hidden_heads
            self.post_cif_layers = nn.ModuleList([
                WhisperEncoderLayer(WhisperConfig(**config_dict)) for _ in range(num_post_cif_layers)
            ])
        self.token_embed_proj = nn.Linear(whisper_config.d_model, out_dim)
        self.lm_head = nn.Linear(whisper_config.d_model, vocab_size, bias=False)

    def forward_cif(self, hidden_states, alphas, num_tokens=None, threshold=1.0):
        device = hidden_states.device
        B, T, H = hidden_states.size()

        if num_tokens is None:
            num_tokens = torch.round(alphas.sum(-1)).int()
        else:
            num_tokens = num_tokens.clone()
        num_tokens[num_tokens < 1] = 1
        max_tokens = num_tokens.max()

        attention_mask = length_to_attention_mask(num_tokens)

        # loop vars
        integrate = torch.zeros([B], device=device)  # accumulated alpha value that hasn't benen fired yet
        remainds = torch.zeros([B], device=device)  # reamining alpha value from recent firing
        token_index = torch.zeros([B], dtype=torch.long, device=device)  # num of fires that has happened

        weights = torch.zeros((B, max_tokens, T), device=device)
        for t in range(T):
            if t > 0:
                weights[:, :, t - 1].scatter_add_(dim=1, index=token_index.unsqueeze(1), src=remainds.unsqueeze(1))

            alpha = alphas[:, t]
            alpha_needed = 1 - integrate
            integrate += alpha
            ready_to_fire = integrate >= threshold

            while True:  # allow repeated firing if integrate > threshold
                integrate = torch.where(ready_to_fire, integrate - 1, integrate)
                alpha_integrated = torch.where(ready_to_fire, alpha_needed, alpha)

                weights[:, :, t].scatter_(dim=1, index=token_index.unsqueeze(1), src=alpha_integrated.unsqueeze(1))
                remainds = alpha - alpha_integrated

                token_index = token_index + ready_to_fire.type_as(token_index)
                token_index = torch.minimum(token_index, num_tokens - 1)

                alpha = remainds
                alpha_needed = 1
                ready_to_fire = integrate >= threshold
                if not ready_to_fire.any():
                    break
                else:
                    pass

        cif_weights = weights.type_as(hidden_states).bmm(hidden_states)

        return cif_weights, attention_mask

    def get_alphas(self, hidden_states, attention_mask):
        alphas = hidden_states[:, :, -1]  ##  B x T x D -> B x T
        alphas = torch.sigmoid(alphas)
        alphas = alphas * attention_mask.float()
        return alphas

    def resize(self, alphas, num_tokens):
        device = alphas.device

        # sum
        orig_alphas_sum = alphas.sum(-1)

        B, T = alphas.size()

        # scaling
        scaled_alphas = alphas * (num_tokens.float() / orig_alphas_sum)[:, None].repeat(1, T)

        return scaled_alphas, orig_alphas_sum

    def forward(self, hidden_states, attention_mask, num_tokens=None):
        encoder_hidden_states = hidden_states
        if self.num_pre_cif_layers > 0:
            # hidden_states = hidden_states.transpose(0,1) # B x T x C -> T x B x C
            for layer in self.pre_cif_layers:
                hidden_states = layer(hidden_states, None, None, False)[0]
            # hidden_states = hidden_states.transpose(0,1) # T x B x C -> B x T x C
        else:
            hidden_states = self.pre_cif_layer(hidden_states)

        alphas = self.get_alphas(hidden_states, attention_mask)
        if self.training:
            assert num_tokens is not None
        else:
            num_tokens = torch.round(alphas.sum(-1)).int()
            num_tokens[num_tokens < 1] = 1

        alphas, alphas_sum = self.resize(alphas, num_tokens)

        hidden_states, attention_mask = self.forward_cif(hidden_states[:, :, :-1], alphas, num_tokens)
        hidden_states = self.cif_proj(hidden_states)

        if self.num_post_cif_layers > 0:
            # hidden_states = hidden_states.transpose(0, 1)
            layer_masking = (length_to_4d_attention_mask(attention_mask.sum(dim=-1))).to(dtype=hidden_states.dtype)
            for layer in self.post_cif_layers:
                hidden_states = layer(hidden_states, layer_masking, None, False)[0]
            # hidden_states = hidden_states.transpose(0, 1)

        logits = self.lm_head(hidden_states)
        hidden_states = self.token_embed_proj(hidden_states)

        return hidden_states, attention_mask, logits, alphas, alphas_sum


