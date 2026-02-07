import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modeling.modules.blocks import modulate, FinalLayer, SwiGLUFFN, RMSNorm
import math
import numpy as np

try:
    from omegaconf import ListConfig
except ImportError:
    ListConfig = None


# This block is used to build adaMLP
class ResBlock(nn.Module):
    """Residual block with AdaLN modulation and SwiGLU activation."""

    def __init__(self, channels, norm_layer=RMSNorm):
        super().__init__()
        self.channels = channels
        self.in_ln = norm_layer(channels)
        # Always use SwiGLU with mlp_ratio=4.0
        self.mlp = SwiGLUFFN(in_features=channels, hidden_features=int(2/3 * int(channels * 4.0)))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, attn_mask=None, c=None):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class GaussianFourierEmbedding(nn.Module):
    """Gaussian Fourier Embedding for timesteps."""

    def __init__(self, hidden_size: int, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale
        self.W = nn.Parameter(torch.normal(0, self.scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        with torch.no_grad():
            W = self.W
        t = t[:, None] * W[None, :] * 2 * torch.pi
        t_embed = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        t_embed = self.mlp(t_embed)
        return t_embed


class MaskBitModelingHead(nn.Module):
    """Mask Bit Modeling head for token prediction.

    Simplified version with hardcoded best-performing configurations:
    - Always uses SwiGLU activation and RMSNorm
    - Logit-normal timestep sampling with mean=0.0, std=0.8
    - Identity timestep-to-mask-ratio mapping
    - Binary quantization (target_codebook_size=2)
    """

    def __init__(self, target_codebook_size=2, num_layers=3, width=2048, seq_len=16):
        super(MaskBitModelingHead, self).__init__()

        self.num_layers = num_layers
        self.width = width
        self.seq_len = seq_len

        # Always use RMSNorm and SwiGLU
        norm_layer = RMSNorm

        # Input embedding and projection
        self.input_embed = nn.Embedding(target_codebook_size + 1, math.ceil(self.width / self.seq_len))
        self.input_proj = nn.Linear(math.ceil(self.width / self.seq_len) * self.seq_len, self.width, bias=True)
        self.ln_pre = norm_layer(self.width)

        # Transformer blocks with SwiGLU (mlp_ratio=4.0 hardcoded)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResBlock(
                channels=self.width,
                norm_layer=norm_layer,
            ))

        # Output projection
        self.output_embed = nn.Linear(self.width, self.seq_len * target_codebook_size, bias=True)

        self.mask_token_id = target_codebook_size
        self.target_codebook_size = target_codebook_size

        self.apply(self._init_weights)

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.t_embedder = GaussianFourierEmbedding(self.width)
        self.adaln_before_head = FinalLayer(self.width, norm_layer)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            m.weight.data = nn.init.trunc_normal_(m.weight.data, mean=0.0, std=0.02)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def _sample_timesteps(self, batch_size, device):
        """Sample timesteps using logit-normal distribution.

        Args:
            batch_size: Number of timesteps to sample
            device: Device to place the timesteps on

        Returns:
            Tensor of shape (batch_size,) with values in [0, 1]
        """
        # Sample from N(0.0, 0.8²) and apply sigmoid
        z = torch.randn((batch_size,), device=device) * 0.8 + 0.0
        return torch.sigmoid(z)

    def _map_timesteps_to_mask_ratio(self, timesteps):
        """Map timesteps to mask ratios using identity mapping.

        In diffusion convention: t=0 means noisy (mask_ratio=1.0), t=1 means clean (mask_ratio=0.0)

        Args:
            timesteps: Tensor of timesteps in [0, 1]

        Returns:
            Tensor of mask ratios where 1.0 means fully masked, 0.0 means no mask
        """
        # Direct inversion: t=0 → mask_ratio=1.0, t=1 → mask_ratio=0.0
        mask_ratio = torch.clamp(1.0 - timesteps, min=1 / self.seq_len, max=1.)
        return mask_ratio

    def masking_input_tokens(self, input_tokens):
        """Mask input tokens for training.

        Args:
            input_tokens: Input tokens to mask [B, seq_len]

        Returns:
            masked_tokens, masks, mask_ratio
        """
        batch_size, seq_len = input_tokens.shape
        assert seq_len == self.seq_len, f"Input tokens length {seq_len} does not match expected {self.seq_len}."
        device = input_tokens.device

        timesteps = self._sample_timesteps(batch_size, device)
        mask_ratio = self._map_timesteps_to_mask_ratio(timesteps)

        num_token_masked = (seq_len * mask_ratio).round().clamp(min=1)
        batch_randperm = torch.rand(batch_size, seq_len, device=device).argsort(dim=-1)
        masks = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

        # Always use mask token (not random tokens)
        mask_token_fill = torch.full_like(input_tokens, self.mask_token_id)
        masked_tokens = torch.where(masks, mask_token_fill, input_tokens)
        return masked_tokens, masks, mask_ratio

    def forward_fn(self, masked_input_ids, conditions, mask_ratio):
        inputs = self.input_embed(masked_input_ids)
        inputs = rearrange(inputs, 'b l c -> b (l c)')
        inputs = self.input_proj(inputs)

        t_emb = self.t_embedder(mask_ratio).unsqueeze(1)
        y_emb = conditions
        s = F.silu(t_emb + y_emb).squeeze(1)

        x = self.ln_pre(inputs)
        for i in range(self.num_layers):
            x = self.transformer[i](x, c=s)

        x = self.adaln_before_head(x, s)
        x = self.output_embed(x)
        x = rearrange(x, 'b (s c) -> b s c', s=self.seq_len, c=self.target_codebook_size)
        return x

    def forward(self, target, conditions):
        """Forward pass through MBM head.

        Args:
            target: Target tokens (clean labels for loss computation)
            conditions: Latent conditions from BAR
        """
        # Mask input tokens
        masked_inputs, masks, mask_ratio = self.masking_input_tokens(target)
        predictions = self.forward_fn(masked_inputs, conditions, mask_ratio)

        with torch.cuda.amp.autocast(enabled=False):
            loss = self.loss_fn(rearrange(predictions.float(), 'b l c -> b c l'), target)
            masks = masks.to(loss).float()
            # Masked tokens weighted at 1.0, unmasked at 0.1
            loss_weights = (1.0 - masks) * 0.1 + masks
            loss = (loss * loss_weights).sum() / (loss_weights.sum() + 1e-8)
            return loss


    def _sample_tokens(self, logits, annealed_temp, add_gumbel_noise):
        """Sample tokens from logits with gumbel noise."""
        sampled_ids = add_gumbel_noise(logits, annealed_temp).argmax(dim=-1)
        sampled_logits = torch.squeeze(
            torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)
        return sampled_ids, sampled_logits

    def _compute_masking(self, sampled_logits, next_mask_ratio, annealed_temp, add_gumbel_noise,
                        device, is_mask):
        """Compute which tokens to mask for next step."""
        # Compute mask length (use round to match training behavior)
        mask_len = torch.Tensor([np.round(self.seq_len * next_mask_ratio)]).to(device)
        # Consider only currently masked positions
        mask_len = torch.maximum(
            torch.Tensor([1]).to(device),
            torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1, mask_len)
        )[0].squeeze()

        # Compute confidence and masking threshold
        confidence = add_gumbel_noise(sampled_logits, annealed_temp)
        sorted_confidence, _ = torch.sort(confidence, axis=-1)
        cut_off = sorted_confidence[:, mask_len.long() - 1:mask_len.long()]
        return confidence <= cut_off

    @torch.no_grad()
    def sample(self, conditions, guidance_scale=3.0, randomize_temperature=4.5, tokens_allocation=[4, 4, 4, 4], use_cfg=False):
        """Sample tokens using MBM iterative decoding.

        Args:
            conditions: Latent conditions. When CFG is used (use_cfg=True),
                       conditions are doubled: [batch_size*2, 1, width] where first half
                       is conditional, second half is unconditional.
            guidance_scale: CFG guidance scale. Used when use_cfg=True.
            randomize_temperature: Temperature for sampling (default: 4.5)
            tokens_allocation: Token unmasking schedule (default: [2, 2, 5, 7]).
                             Example: [2, 2, 5, 7] means 4 steps unmasking 2, 2, 5, 7 tokens.
                             Must sum to self.seq_len and be non-decreasing.
            use_cfg: Whether to apply CFG. If True, conditions should be doubled.
        """
        device = conditions.device

        if use_cfg:
            # CFG is active, conditions are doubled [batch_size*2, 1, width]
            batch_size = conditions.shape[0] // 2
        else:
            batch_size = conditions.shape[0]

        # Helper functions for gumbel noise
        def log(t, eps=1e-20):
            return torch.log(t.clamp(min=eps))

        def gumbel_noise(t):
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))

        def add_gumbel_noise(t, temperature):
            return t + temperature * gumbel_noise(t)

        # Convert OmegaConf ListConfig to plain list if needed
        if ListConfig is not None and isinstance(tokens_allocation, ListConfig):
            tokens_allocation = list(tokens_allocation)

        # Validate tokens_allocation
        if not isinstance(tokens_allocation, (list, tuple)):
            raise ValueError(f"tokens_allocation must be a list or tuple, got {type(tokens_allocation)}")
        if len(tokens_allocation) == 0:
            raise ValueError("tokens_allocation must have at least one element")
        if any(t <= 0 for t in tokens_allocation):
            raise ValueError(f"tokens_allocation values must be positive, got {tokens_allocation}")
        if sum(tokens_allocation) != self.seq_len:
            raise ValueError(
                f"tokens_allocation must sum to seq_len ({self.seq_len}), "
                f"but got sum={sum(tokens_allocation)}, allocation={tokens_allocation}"
            )
        # Check non-decreasing
        for i in range(len(tokens_allocation) - 1):
            if tokens_allocation[i+1] < tokens_allocation[i]:
                raise ValueError(
                    f"tokens_allocation must be non-decreasing, but got {tokens_allocation}"
                )

        num_sample_steps = len(tokens_allocation)
        cumulative_tokens = [0] + [sum(tokens_allocation[:i+1]) for i in range(len(tokens_allocation))]

        # Initialize ids with mask tokens
        ids = torch.full((batch_size, self.seq_len), self.mask_token_id, device=device)

        # Sampling loop
        for step in range(num_sample_steps):
            # Compute mask ratios
            mask_ratio = 1.0 - (cumulative_tokens[step] / self.seq_len)
            next_mask_ratio = 1.0 - (cumulative_tokens[step + 1] / self.seq_len)

            # Temperature annealing: from randomize_temperature to 0.0
            ratio = step / num_sample_steps
            annealed_temp = randomize_temperature * (1.0 - ratio)

            # Prepare mask ratio tensor
            mask_ratio_t = torch.tensor(mask_ratio, dtype=conditions.dtype, device=device)

            # Forward pass with or without CFG
            if use_cfg:
                # Apply CFG: double ids to match doubled conditions
                mask_ratio_t = mask_ratio_t.view(1).repeat(batch_size * 2)
                logits = self.forward_fn(torch.cat([ids, ids], dim=0), conditions, mask_ratio_t)
                # Split and apply CFG
                cond_logits, uncond_logits = torch.split(logits, batch_size, dim=0)
                logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
            else:
                mask_ratio_t = mask_ratio_t.view(1).repeat(batch_size)
                logits = self.forward_fn(ids, conditions, mask_ratio_t)

            # Sample tokens
            sampled_ids, sampled_logits = self._sample_tokens(logits, annealed_temp, add_gumbel_noise)

            # Only update masked positions
            is_mask = (ids == self.mask_token_id)
            sampled_ids = torch.where(is_mask, sampled_ids, ids)
            sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()

            # Update ids for next step
            if step == num_sample_steps - 1:
                ids = sampled_ids
            else:
                # Compute masking for next step
                masking = self._compute_masking(
                    sampled_logits, next_mask_ratio, annealed_temp, add_gumbel_noise,
                    device, is_mask
                )
                ids = torch.where(masking, self.mask_token_id, sampled_ids)

        return ids


