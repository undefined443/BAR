import torch.nn as nn
import json
from omegaconf import OmegaConf
from pathlib import Path

from .modules import BaseModel, SigLIP2Encoder, SigLIP2Decoder
from .modules.blocks import RMSNorm
from .quantizer import FSQ


class BAR_FSQ(BaseModel):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config

        # Initialize decoder first, then apply weight initialization
        self.decoder = SigLIP2Decoder(config)
        self.apply(self._init_weights)

        # Load encoder after weight initialization to preserve pretrained weights
        self.encoder = SigLIP2Encoder(config)

        # Only use FSQ quantizer
        self.quantize = FSQ(
            in_channel=self.encoder.width,
            out_channel=self.decoder.width,
            token_size=config.model.vq_model.token_size,
            config=config,
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, "w") as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """Initialize the weights.

        Args:
            module: torch.nn.Module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data, mean=0.0, std=0.02
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data, mean=0.0, std=0.02
            )
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)

    def encode(self, x):
        """Encode images to quantized tokens.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            z_quantized: Quantized tokens (B, N, C)
            result_dict: Dictionary containing quantization results and clip_gt
        """
        z, clip_gt = self.encoder(pixel_values=x, return_clip_gt=True)
        z_quantized, result_dict = self.quantize(z)
        result_dict["clip_gt"] = clip_gt
        return z_quantized, result_dict

    def decode(self, z_quantized):
        """Decode quantized tokens to images.

        Args:
            z_quantized: Quantized tokens (B, N, C)

        Returns:
            decoded: Reconstructed images (B, 3, H, W)
            clip_pred: CLIP predictions for semantic feature reconstruction
        """
        decoded, clip_pred = self.decoder(z_quantized)
        return decoded, clip_pred

    def decode_tokens(self, tokens):
        """Decode discrete token indices to images.

        Args:
            tokens: Discrete token indices (B, N)

        Returns:
            decoded: Reconstructed images (B, 3, H, W)
        """
        tokens = tokens.squeeze(1)
        batch, seq_len = tokens.shape  # B x N
        z_quantized = self.quantize.get_codebook_entry(
            tokens.reshape(batch, -1)
        ).reshape(batch, -1, self.decoder.width)
        decoded = self.decode(z_quantized)[0]
        return decoded

    def forward(self, input):
        """Forward pass: encode -> quantize -> decode.

        Args:
            input: Input images (B, 3, H, W)

        Returns:
            decoded: Reconstructed images (B, 3, H, W)
            result_dict: Dictionary containing quantization results, CLIP predictions and ground truth
        """
        z_quantized, result_dict = self.encode(input)
        decoded, clip_pred = self.decode(z_quantized)
        result_dict["clip_pred"] = clip_pred
        return decoded, result_dict
