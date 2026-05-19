"""SigLIP2 vision encoder + CLIP text tokenizer for caption training."""

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from transformers import AutoModel, CLIPTokenizer, SiglipImageProcessor

from .modules import SigLIP2Decoder


class BAR_FSQ(nn.Module):
    """Wrapper for CLIP text tokenizer + SigLIP2 vision encoder."""

    def __init__(self, config: DictConfig | dict) -> None:

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config

        self._decoder = SigLIP2Decoder(config)

        self._tokenizer = CLIPTokenizer.from_pretrained(config.model.vq_model.tokenizer)
        self._text_seq_len = config.model.generator.text_seq_len

        full_model = AutoModel.from_pretrained(config.model.vq_model.encoder)
        self._encoder = full_model.vision_model

        # Freeze the SigLIP2 model
        self._encoder.eval()
        self._encoder.requires_grad_(False)
        self._encoder.head.requires_grad_(False)

        del full_model

        crop_size = config.dataset.preprocessing.crop_size
        self._image_processor = SiglipImageProcessor(
            size={"height": crop_size, "width": crop_size}, do_resize=False
        )

        self._token_bitmap_lut = self._build_token_bitmap_lut()

    def encode(
        self, texts: list[str] | str, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode texts as images and images to embeddings.

        Args:
            texts: List of text strings or single text string.
            images: (B, 3, H, W) tensor in [-1, 1].

        Returns:
            Tuple of (token_bits, image_embeddings).
            token_bits: (B, L*D) tensor where L=text_seq_len and D is flattened image dimension.
            image_embeddings: (B, num_patches, hidden_size) from SigLIP2 vision model.
        """
        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            inputs = self._tokenizer(
                texts,
                padding="max_length",
                max_length=self._text_seq_len,
                return_tensors="pt",
                truncation=True,
            )
            # Look up pre-rendered bitmaps: (B, text_seq_len) -> (B, text_seq_len * D)
            token_ids = inputs["input_ids"]
            token_bits = (
                self._token_bitmap_lut[token_ids]
                .to(images.device)
                .long()
                .reshape(len(token_ids), -1)
            )

            # Rescale from [-1, 1] to [0, 1] before passing to SiglipImageProcessor
            images_rescaled = (images * 0.5 + 0.5).clamp(0, 1)
            processed_images = self._image_processor(
                images=images_rescaled, return_tensors="pt", do_rescale=False
            )["pixel_values"].to(images.device)
            outputs = self._encoder(pixel_values=processed_images)
            image_embeddings = outputs.last_hidden_state

        return token_bits, image_embeddings

    def decode_tokens(self, token_bits: torch.Tensor) -> list[Image.Image]:
        """Convert (B, L*D) token image tensor back to PIL images.

        Args:
            token_bits: (B, L*D) tensor where L=text_seq_len and D is flattened image dimension.

        Returns:
            List of PIL Images.
        """
        B = token_bits.shape[0]
        char_image_size = self.config.model.vq_model.get("char_image_size", 14)
        max_token_length = self.config.model.vq_model.get("max_token_length", 15)

        # Reshape from (B, L*D) to (B, 1, char_image_size, W)
        tensor_reshaped = token_bits.view(B, 1, -1, char_image_size * max_token_length)

        images = [
            to_pil_image((t * 255).cpu().byte()).convert("1") for t in tensor_reshaped
        ]
        return images

    def _build_token_bitmap_lut(self) -> torch.Tensor:
        """Pre-render all vocabulary tokens as bitmaps for O(1) encode lookup.

        Called once at __init__. Eliminates per-batch PIL rendering by building a
        (vocab_size, D) uint8 tensor where D = char_image_size * max_token_length *
        char_image_size. encode() indexes into this table with token_ids directly.

        Returns:
            Tensor of shape (vocab_size, D) dtype=uint8.
        """
        max_token_length = self.config.model.vq_model.get("max_token_length", 15)
        char_image_size = self.config.model.vq_model.get("char_image_size", 14)

        terminus_font = Path("/usr/share/fonts/opentype/terminus/terminus-normal.otb")
        if not terminus_font.exists():
            raise FileNotFoundError(
                "Terminus font not found. Install it with: apt install fonts-terminus-otb"
            )
        # Load font once for all tokens instead of per character per call.
        font = ImageFont.truetype(str(terminus_font), size=12)

        vocab_size = len(self._tokenizer)
        all_tokens = self._tokenizer.convert_ids_to_tokens(list(range(vocab_size)))

        D = char_image_size * max_token_length * char_image_size
        lut = torch.zeros(vocab_size, D, dtype=torch.uint8)

        for idx, token in enumerate(all_tokens):
            if token is None:
                continue
            token = token.ljust(max_token_length)[:max_token_length]
            img = Image.new(
                "1", (max_token_length * char_image_size, char_image_size), color=1
            )
            draw = ImageDraw.Draw(img)
            for i, char in enumerate(token):
                draw.text(
                    (i * char_image_size + char_image_size // 2, char_image_size // 2),
                    char,
                    fill=0,
                    font=font,
                    anchor="mm",
                )
            lut[idx] = pil_to_tensor(img).view(-1)

        return lut

    def forward(self, input: list[str] | str) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(input)
