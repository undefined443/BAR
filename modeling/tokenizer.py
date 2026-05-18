"""SigLIP2 vision encoder + CLIP text tokenizer for caption training."""

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from transformers import AutoModel, CLIPTokenizer, SiglipImageProcessor

from .modules import SigLIP2Decoder


class BAR_FSQ(nn.Module):
    """Wrapper for CLIP text tokenizer + SigLIP2 vision encoder."""

    def __init__(self, config):

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

    def encode(self, texts, images):
        """Encode texts as images and images to embeddings.

        Args:
            texts: List of text strings or single text string.
            images: (B, 3, H, W) tensor in [-1, 1].

        Returns:
            Tuple of (token_bits, image_embeddings).
            token_bits: (B, L, D) tensor where L=text_seq_len and D is flattened image dimension.
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
            token_ids = inputs["input_ids"].to(images.device)
            text_images = [self._token_ids_to_images(tid.tolist()) for tid in token_ids]
            images_tensor = [
                pil_to_tensor(img).long().to(images.device).view(self._text_seq_len, -1)
                for img in text_images
            ]
            token_bits = torch.stack(images_tensor)

            # Rescale from [-1, 1] to [0, 1] before passing to SiglipImageProcessor
            images_rescaled = (images * 0.5 + 0.5).clamp(0, 1)
            processed_images = self._image_processor(
                images=images_rescaled, return_tensors="pt", do_rescale=False
            )["pixel_values"].to(images.device)
            outputs = self._encoder(pixel_values=processed_images)
            image_embeddings = outputs.last_hidden_state

        return token_bits, image_embeddings

    def decode_tokens(self, token_bits):
        """Convert (B, L, D) token image tensor back to PIL images.

        Args:
            token_bits: (B, L, D) tensor where L=text_seq_len and D is flattened image dimension.

        Returns:
            List of PIL Images.
        """
        B, _, _ = token_bits.shape
        char_image_size = self.config.model.vq_model.get("char_image_size", 14)

        # Reshape from (B, L, D) to (B, 1, char_image_size, W)
        tensor_reshaped = token_bits.view(B, 1, char_image_size, -1)

        # Convert to PIL Image
        images = [to_pil_image(t.cpu().byte()) for t in tensor_reshaped]
        return images

    def forward(self, input):
        return self.encode(input)

    @staticmethod
    def _character_to_image(char: str, img_size: int = 14) -> Image.Image:
        """Convert a character to a PIL Image.

        Args:
            char: A single character string
            img_size: Size of the output image in pixels (default: 14)

        Raises:
            AssertionError: If input is not a string or not a single character
            FileNotFoundError: If Terminus font is not installed

        Returns:
            PIL Image with the character rendered
        """
        assert isinstance(char, str), "Input must be a string."
        assert len(char) == 1, "Input must be a single character."

        terminus_font = Path("/usr/share/fonts/opentype/terminus/terminus-normal.otb")
        if not terminus_font.exists():
            raise FileNotFoundError(
                "Terminus font not found. Install it with: apt install fonts-terminus-otb"
            )

        img = Image.new("1", (img_size, img_size), color=1)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(terminus_font, size=12)
        draw.text((img_size // 2, img_size // 2), char, fill=0, font=font, anchor="mm")

        return img

    @staticmethod
    def _word_to_image(
        word: str, max_length: int | None = None, img_size: int = 14
    ) -> Image.Image:
        """Convert a word string to an image with each character rendered.

        Args:
            word: A word string
            max_length: If set, pad or truncate word to this length (pad with spaces)
            img_size: Size of each character image in pixels (default: 14)

        Returns:
            PIL Image with all characters concatenated horizontally
        """
        if max_length is not None:
            if len(word) < max_length:
                word = word.ljust(max_length)
            else:
                word = word[:max_length]

        char_images = [
            BAR_FSQ._character_to_image(char, img_size=img_size) for char in word
        ]

        char_width = char_images[0].width
        char_height = char_images[0].height
        width = len(word) * char_width
        img = Image.new("1", (width, char_height), color=1)
        for i, char_img in enumerate(char_images):
            img.paste(char_img, (i * char_width, 0))

        return img

    @staticmethod
    def _words_to_image(
        tokens: list[str], max_length: int | None = None, img_size: int = 14
    ) -> Image.Image:
        """Convert a list of tokens to a concatenated image.

        Args:
            tokens: List of token strings
            max_length: If set, pad/truncate each token to this length
            img_size: Size of each character image in pixels (default: 14)

        Returns:
            PIL Image with all tokens concatenated horizontally
        """
        word_images = [
            BAR_FSQ._word_to_image(token, max_length=max_length, img_size=img_size)
            for token in tokens
        ]

        total_width = sum(img.width for img in word_images)
        height = word_images[0].height
        img = Image.new("1", (total_width, height), color=1)

        x_offset = 0
        for word_img in word_images:
            img.paste(word_img, (x_offset, 0))
            x_offset += word_img.width

        return img

    def _token_ids_to_images(self, token_ids: list[int]) -> Image.Image:
        """Convert token IDs to image representation by rendering tokens as characters.

        Args:
            token_ids: List of token IDs (length: text_seq_len).

        Returns:
            PIL Image with all tokens rendered as characters and concatenated horizontally.
            Shape: (char_image_size, text_seq_len * max_token_length * char_image_size)
        """
        max_token_length = self.config.model.vq_model.get("max_token_length")
        char_image_size = self.config.model.vq_model.get("char_image_size", 14)
        tokens = self._tokenizer.convert_ids_to_tokens(token_ids)
        images = BAR_FSQ._words_to_image(
            tokens, max_length=max_token_length, img_size=char_image_size
        )
        return images
