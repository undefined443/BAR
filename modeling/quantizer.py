import torch
import torch.nn as nn
from torch.amp import autocast


class FSQ(nn.Module):
    """Finite Scalar Quantization.

    Quantizes continuous features into discrete levels per channel without a learned codebook.
    Binary quantization (levels_per_channel=2) is always used.
    """

    def __init__(
        self,
        in_channel=None,
        out_channel=None,
        token_size=16,  # equals to number of channels, as FSQ applies quant per channel
        config=None,
    ):
        super().__init__()
        self.config = config
        self.token_size = token_size
        self.levels_per_channel = 2  # Binary quantization is always used

        self.in_proj = (
            nn.Linear(in_channel, token_size)
            if in_channel is not None
            else nn.Identity()
        )
        self.out_proj = (
            nn.Linear(token_size, out_channel)
            if out_channel is not None
            else nn.Identity()
        )

        # Quantize to 2 levels per channel
        levels = [self.levels_per_channel] * token_size
        assert (
            config.model.vq_model.codebook_size == self.levels_per_channel**token_size
        ), (
            f"codebook_size {config.model.vq_model.codebook_size} != "
            f"levels_per_channel {self.levels_per_channel} ** token_size {token_size}"
        )

        _levels = torch.tensor(levels, dtype=torch.int64)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int64
        )
        self.register_buffer("_basis", _basis, persistent=False)

        self.codebook_size = config.model.vq_model.codebook_size

    def round_ste(self, z):
        """Round with straight-through gradients."""
        zhat = z.round()
        return z + (zhat - z).detach()

    def bound(self, z, eps: float = 1e-3):
        """Bound z to quantization range.

        For binary quantization (levels=2):
        - Maps to range [-1, 0] before rounding
        - After rounding: -1 or 0
        """
        half_l = (self._levels - 1) * (1 + eps) / 2  # For levels=2: 0.5015
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)  # For levels=2: 0.5
        shift = (offset / half_l).atanh()

        # Normalize to (-1, 1)
        normalized_z = (z + shift).tanh()

        return normalized_z * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize z to discrete levels.

        For binary quantization (levels=2):
        - bound() maps to [-1, 0] range
        - round_ste() rounds to {-1, 0}
        - Final mapping transforms {-1, 0} -> {-1, +1}
        """
        # Bound and round: results in {-1, 0} for levels=2
        quantized = self.round_ste(self.bound(z))

        # For binary quantization: map {-1, 0} to {-1, +1}
        # Formula: (quantized / half_width) * 2 + 1
        # -1 -> (-1/1)*2 + 1 = -1
        #  0 -> (0/1)*2 + 1 = +1
        half_width = self._levels // 2  # For levels=2: half_width=1
        return (quantized / half_width) * 2 + 1

    def _scale_and_shift(self, zhat_normalized):
        """Scale normalized codes {-1, +1} to integer levels {0, 1}.

        For binary quantization: {-1, +1} -> {0, 1}
        """
        # Map {-1, +1} to {0, 1}
        return (zhat_normalized + 1) / 2

    def _scale_and_shift_inverse(self, zhat):
        """Inverse of _scale_and_shift: {0, 1} -> {-1, +1}.

        For binary quantization: {0, 1} -> {-1, +1}
        """
        # Sanity check: zhat should only contain 0 or 1
        assert torch.all((zhat == 0) | (zhat == 1)), (
            f"Expected zhat to contain only 0 or 1, but got values in range [{zhat.min()}, {zhat.max()}]"
        )

        # Map {0, 1} to {-1, +1}
        return zhat.float() * 2 - 1

    def get_codebook_entry(self, indices):
        """Get quantized codes from indices.

        Args:
            indices: (B, L*C) flattened indices per level

        Returns:
            codes: (B, L, C) quantized codes
        """
        indices_per_level = indices.reshape(indices.shape[0], -1, self.token_size)
        codes = self.indices_to_codes_per_level(indices_per_level)
        codes = self.out_proj(codes)
        return codes

    def codes_to_indices_per_level(self, zhat_normalized):
        """Convert normalized codes to per-level indices.

        Args:
            zhat_normalized: (B, L, C) float codes in [-1, 1]

        Returns:
            codes: (B, L, C) integer codes in [0, levels_per_channel-1]
        """
        levels_float = self._scale_and_shift(zhat_normalized)
        codes = torch.round(levels_float).to(torch.int64)
        return torch.clamp(codes, 0, self.levels_per_channel - 1)

    def indices_to_codes_per_level(self, per_level_codes):
        """Convert per-level indices to normalized codes."""
        per_level_codes = per_level_codes.to(torch.int64)
        return self._scale_and_shift_inverse(per_level_codes)

    @autocast(device_type="cuda", enabled=False)
    def forward(self, z, condition=None):
        """Forward pass: quantize continuous features with binary quantization.

        Args:
            z: (B, L, C) continuous features
            condition: (unused, for compatibility)

        Returns:
            z_quantized: (B, L, C) quantized features
            result_dict: Dictionary with quantization info
        """
        z = z.float()
        z = self.in_proj(z)

        assert z.shape[-1] == self.token_size
        # Quantize: each channel in {-1, +1}
        z_quantized = self.quantize(z)

        # Convert to per-level indices for later decoding
        min_encoding_indices = self.codes_to_indices_per_level(z_quantized)

        z_quantized = self.out_proj(z_quantized)

        # Return zero losses (no commitment loss, no entropy loss)
        result_dict = dict(min_encoding_indices=min_encoding_indices)

        return z_quantized, result_dict
