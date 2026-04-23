"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(
        self,
        text_num_embeds,
        text_dim,
        conv_layers=0,
        conv_mult=2,
        use_duration_condition=False,
        duration_log_scale=100.0,
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token
        self.use_duration_condition = use_duration_condition
        self.duration_log_scale = duration_log_scale
        self._last_duration_metrics = {}
        if self.use_duration_condition:
            self.content_duration_mlp = nn.Sequential(
                nn.Linear(1, text_dim),
                nn.SiLU(),
                nn.Linear(text_dim, text_dim),
            )
            self.pause_duration_mlp = nn.Sequential(
                nn.Linear(1, text_dim),
                nn.SiLU(),
                nn.Linear(text_dim, text_dim),
            )
            self.alpha_content = nn.Parameter(torch.zeros(1))
            self.alpha_pause = nn.Parameter(torch.zeros(1))

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(
        self,
        text: int["b nt"],
        seq_len,
        drop_text=False,
        token_durations=None,
        token_duration_mask=None,
    ):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        if self.use_duration_condition and token_durations is not None:
            token_durations = token_durations[:, :seq_len].to(text.device, text.dtype)
            if token_durations.shape[1] < seq_len:
                pad_right = seq_len - token_durations.shape[1]
                token_durations = F.pad(token_durations, (0, 0, 0, pad_right), value=0.0)
            if token_durations.ndim == 2:
                token_durations = token_durations.unsqueeze(-1)
            content_duration = token_durations[..., 0:1]
            pause_after = token_durations[..., 1:2] if token_durations.shape[-1] > 1 else torch.zeros_like(content_duration)
            if token_duration_mask is not None:
                token_duration_mask = token_duration_mask[:, :seq_len].to(text.device, text.dtype)
                if token_duration_mask.ndim == 2:
                    token_duration_mask = token_duration_mask.unsqueeze(-1)
                if token_duration_mask.shape[-1] == 1:
                    token_duration_mask = torch.cat([token_duration_mask, token_duration_mask], dim=-1)
                elif token_duration_mask.shape[-1] > 2:
                    token_duration_mask = token_duration_mask[..., :2]
                if token_duration_mask.shape[1] < seq_len:
                    pad_right = seq_len - token_duration_mask.shape[1]
                    token_duration_mask = F.pad(token_duration_mask, (0, 0, 0, pad_right), value=0.0)
            else:
                content_mask_default = torch.ones_like(content_duration)
                pause_mask_default = (
                    torch.ones_like(content_duration)
                    if token_durations.shape[-1] > 1
                    else torch.zeros_like(content_duration)
                )
                token_duration_mask = torch.cat([content_mask_default, pause_mask_default], dim=-1)

            content_mask = token_duration_mask[..., 0:1]
            pause_mask = token_duration_mask[..., 1:2]
            content_in = torch.log1p(content_duration * self.duration_log_scale)
            pause_in = torch.log1p(pause_after * self.duration_log_scale)
            zero_content_in = torch.zeros_like(content_in)
            zero_pause_in = torch.zeros_like(pause_in)
            # Zero-point correction keeps old checkpoints compatible while ensuring
            # zero duration does not inject a learned default residual.
            content_res = self.content_duration_mlp(content_in) - self.content_duration_mlp(zero_content_in)
            pause_res = self.pause_duration_mlp(pause_in) - self.pause_duration_mlp(zero_pause_in)
            scaled_content_res = self.alpha_content * content_res * content_mask
            scaled_pause_res = self.alpha_pause * pause_res * pause_mask
            duration_res = scaled_content_res + scaled_pause_res
            base_norm = text.norm(dim=-1).mean()
            total_norm = duration_res.norm(dim=-1).mean()
            content_available = content_mask.squeeze(-1) > 0
            pause_available = pause_mask.squeeze(-1) > 0
            content_active = content_available & (content_duration.squeeze(-1) > 0)
            pause_active = pause_available & (pause_after.squeeze(-1) > 0)
            content_res_l2 = scaled_content_res.norm(dim=-1)
            pause_res_l2 = scaled_pause_res.norm(dim=-1)
            zero = total_norm.new_tensor(0.0)

            def masked_mean(values, mask):
                return values[mask].mean() if mask.any() else zero

            content_norm = masked_mean(content_res_l2, content_available)
            pause_norm = masked_mean(pause_res_l2, pause_available)
            self._last_duration_metrics = {
                "duration_gate/alpha_content": self.alpha_content.detach().float().cpu(),
                "duration_gate/alpha_pause": self.alpha_pause.detach().float().cpu(),
                "duration_condition/content_available_ratio": content_mask.mean().detach().float().cpu(),
                "duration_condition/pause_available_ratio": pause_mask.mean().detach().float().cpu(),
                "duration_residual/content_l2": content_norm.detach().float().cpu(),
                "duration_residual/pause_l2": pause_norm.detach().float().cpu(),
                "duration_residual/content_l2_zero_duration": masked_mean(
                    content_res_l2, content_available & ~content_active
                ).detach().float().cpu(),
                "duration_residual/content_l2_nonzero_duration": masked_mean(
                    content_res_l2, content_active
                ).detach().float().cpu(),
                "duration_residual/pause_l2_zero_pause": masked_mean(
                    pause_res_l2, pause_available & ~pause_active
                ).detach().float().cpu(),
                "duration_residual/pause_l2_nonzero_pause": masked_mean(
                    pause_res_l2, pause_active
                ).detach().float().cpu(),
                "duration_residual/total_l2": total_norm.detach().float().cpu(),
                "duration_residual/total_l2_to_base_ratio": (total_norm / (base_norm + 1e-6)).detach().float().cpu(),
            }
            text = text + duration_res
        elif self.use_duration_condition:
            self._last_duration_metrics = {
                "duration_gate/alpha_content": self.alpha_content.detach().float().cpu(),
                "duration_gate/alpha_pause": self.alpha_pause.detach().float().cpu(),
                "duration_residual/content_l2": torch.tensor(0.0),
                "duration_residual/pause_l2": torch.tensor(0.0),
                "duration_residual/content_l2_zero_duration": torch.tensor(0.0),
                "duration_residual/content_l2_nonzero_duration": torch.tensor(0.0),
                "duration_residual/pause_l2_zero_pause": torch.tensor(0.0),
                "duration_residual/pause_l2_nonzero_pause": torch.tensor(0.0),
                "duration_residual/total_l2": torch.tensor(0.0),
                "duration_residual/total_l2_to_base_ratio": torch.tensor(0.0),
            }

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text

    def get_duration_metrics(self):
        return self._last_duration_metrics


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        conv_layers=0,
        long_skip_connection=False,
        duration_condition=False,
        duration_log_scale=100.0,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            conv_layers=conv_layers,
            use_duration_condition=duration_condition,
            duration_log_scale=duration_log_scale,
        )
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)
        self.proj_out_ln_sig = nn.Linear(dim, mel_dim)

    def get_duration_metrics(self):
        return self.text_embed.get_duration_metrics() if hasattr(self.text_embed, "get_duration_metrics") else {}

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        mask: bool["b n"] | None = None,  # noqa: F722
        token_durations=None,
        token_duration_mask=None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        text_embed = self.text_embed(
            text,
            seq_len,
            drop_text=drop_text,
            token_durations=token_durations,
            token_duration_mask=token_duration_mask,
        )
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)
        output_ln_sig = self.proj_out_ln_sig(x)

        return output, output_ln_sig

    def inference(
        self,
        x,     # nosied input audio float['b n d']
        cond,  # masked cond audio float['b n d']
        text,  # text int['b nt']
        time,  # time step float['b'] | float['']
        drop_audio_cond,  # cfg for cond audio
        drop_text,        # cfg for text
        mask=None,  # bool['b n'] | None
        token_durations=None,
        token_duration_mask=None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        x = x.to(cond.dtype)
        time = time.to(cond.dtype)
        t = self.time_embed(time)
        text_embed = self.text_embed(
            text,
            seq_len,
            drop_text=drop_text,
            token_durations=token_durations,
            token_duration_mask=token_duration_mask,
        )
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)
        output_ln_sig = self.proj_out_ln_sig(x)
        snd = torch.randn(output.size()).to(output.device)
        output = output + snd * torch.exp(output_ln_sig)

        return output

    def forward_rl(
        self,
        x,  # nosied input audio float['b n d']
        cond,  # masked cond audio float['b n d']
        text,  # text int['b nt']
        time,  # time step float['b']
        drop_audio_cond,  # cfg for cond audio
        drop_text,   # cfg for text
        mask=None,  # bool['b n'] | None
        token_durations=None,
        token_duration_mask=None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = repeat(time, ' -> b', b=batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        text_embed = self.text_embed(
            text,
            seq_len,
            drop_text=drop_text,
            token_durations=token_durations,
            token_duration_mask=token_duration_mask,
        )
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output_mu = self.proj_out(x)
        output_ln_sig = self.proj_out_ln_sig(x)
        snd = torch.randn(output_mu.size()).to(output_mu.device)
        output = output_mu + snd * torch.exp(output_ln_sig)

        return output, output_mu, output_ln_sig
