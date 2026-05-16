"""SigLIP2 vision encoder + CLIP text tokenizer for caption training."""

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoModel, CLIPTokenizer, SiglipImageProcessor

from .modules import SigLIP2Decoder


class TokenImageCNNEncoder(nn.Module):
    """CNN encoder for token image patches (14×14*max_token_length)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv2d(1, 256, kernel_size=(14, 210), stride=210, padding=0)
        self.act1 = nn.SiLU()
        self.fc1 = nn.Linear(256, 512)
        self.act2 = nn.SiLU()
        self.fc2 = nn.Linear(512, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))  # (B, 256, 1, 1)
        x = x.view(x.shape[0], -1)  # (B, 256)
        x = self.act2(self.fc1(x))  # (B, 512)
        x = self.fc2(x)  # (B, embedding_dim)
        return x


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

        embedding_dim = config.model.vq_model.vision_hidden_size
        self._token_image_encoder = TokenImageCNNEncoder(embedding_dim=embedding_dim)

    def encode(self, texts, images):
        """Encode texts to binary bits and images to embeddings.

        Args:
            texts: List of text strings or single text string.
            images: (B, 3, H, W) tensor in [-1, 1].

        Returns:
            Tuple of (text_token_bits, image_embeddings).
            text_token_bits: (B, L, token_size) binary representation of token IDs.
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
            token_bits = self._token_ids_to_bits(token_ids)

            # Rescale from [-1, 1] to [0, 1] before passing to SiglipImageProcessor
            images_rescaled = (images * 0.5 + 0.5).clamp(0, 1)
            processed_images = self._image_processor(
                images=images_rescaled, return_tensors="pt", do_rescale=False
            )["pixel_values"].to(images.device)
            outputs = self._encoder(pixel_values=processed_images)
            image_embeddings = outputs.last_hidden_state

        return token_bits, image_embeddings

    def decode_tokens(self, token_bits):
        """Decode discrete token indices to text strings.

        Args:
            token_bits: Binary representation of token IDs (B, N * token_size).

        Returns:
            texts: Decoded text strings.
        """
        # Convert bit representation back to token IDs for tokenizer decoding
        token_bits = token_bits.view(token_bits.shape[0], self._text_seq_len, -1)
        token_ids = self._bits_to_token_ids(token_bits)
        texts = self._tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return texts

    def _token_ids_to_bits(self, token_ids, token_size=16):
        """Convert token IDs to binary bits.

        Decomposes each integer token ID into token_size binary bits.

        Args:
            token_ids: (B, L) integer tensor with values in [0, 2^token_size - 1]
            token_size: Number of bits (default: 16)

        Returns:
            bits: (B, L, token_size) with values in {0, 1}
        """
        shifts = torch.arange(token_size, device=token_ids.device)
        # Extract each bit using right shift and bitwise AND
        bits = (token_ids.unsqueeze(-1) >> shifts) & 1
        return bits

    def _bits_to_token_ids(self, bits):
        """Convert binary bits back to token IDs.

        Recomposes token_size binary bits into integer token IDs.

        Args:
            bits: (B, L, token_size) with values in {0, 1}

        Returns:
            token_ids: (B, L) integer tensor
        """
        token_size = bits.shape[-1]
        shifts = torch.arange(token_size, device=bits.device, dtype=torch.long)
        token_ids = (bits.long() << shifts).sum(dim=-1)
        return token_ids

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
                "Terminus font not found. Install it with: apt install fonts-terminus"
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
    def _tokens_to_image(
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

    def _texts_to_images(self, texts: list[str]) -> list[Image.Image]:
        """Convert text strings to image representations via tokenization.

        Args:
            texts: List of text strings to tokenize

        Returns:
            List of PIL Images, one per input text
        """
        max_token_length = self.config.model.vq_model.get("max_token_length")
        char_image_size = self.config.model.vq_model.get("char_image_size", 14)
        tokens_list = [
            self._tokenizer.tokenize(text)[: self._text_seq_len] for text in texts
        ]
        img_list = [
            BAR_FSQ._tokens_to_image(
                tokens, max_length=max_token_length, img_size=char_image_size
            )
            for tokens in tokens_list
        ]

        return img_list

    def _images_to_embedding(
        self,
        images: list[Image.Image],
        char_image_size: int = 14,
        max_token_length: int = 15,
    ) -> torch.Tensor:
        """Convert token images to embeddings using CNN encoder.

        Uses a scanning window of size (char_image_size, char_image_size * max_token_length)
        with stride equal to window width for non-overlapping horizontal patches.

        Args:
            images: List of PIL Images from texts_to_image
            char_image_size: Size of character image in pixels
            max_token_length: Maximum token length (window width = char_image_size * max_token_length)

        Returns:
            Embeddings tensor of shape (B, num_patches, embedding_dim)
        """
        # Convert PIL images to tensors
        image_tensors = []
        for img in images:
            img_tensor = pil_to_tensor(img).float() / 255.0
            image_tensors.append(img_tensor.unsqueeze(0))  # (1, H, W)

        # Stack images (B, 1, H, W)
        batch_images = torch.stack(image_tensors)

        # Get dimensions
        B, C, H, W = batch_images.shape
        scan_h = char_image_size
        scan_w = char_image_size * max_token_length
        stride = scan_w  # Non-overlapping horizontal windows

        # Extract patches using unfold
        patches = batch_images.unfold(2, scan_h, stride).unfold(3, scan_w, stride)
        # patches shape: (B, C, num_h, num_w, scan_h, scan_w)

        B, C, num_h, num_w, _, _ = patches.shape
        # Reshape to (B*num_h*num_w, C, scan_h, scan_w)
        patches = patches.contiguous().view(B * num_h * num_w, C, scan_h, scan_w)

        # CNN encoder
        with torch.no_grad():
            embeddings = self._token_image_encoder(patches)

        # Reshape back to (B, num_patches, embedding_dim)
        embedding_dim = embeddings.shape[-1]
        embeddings = embeddings.view(B, num_h * num_w, embedding_dim)

        return embeddings

    def encode_as_images(self, texts: list[str]) -> torch.Tensor:
        """Encode text strings as images to obtain embeddings.

        Args:
            texts: List of text strings to encode

        Returns:
            Embeddings tensor of shape (B, text_seq_len, embedding_dim)
        """
        images = self._texts_to_images(texts)
        embeddings = self._images_to_embedding(images)
        return embeddings
