"""CLIPEmbedder — compute CLIP embeddings for keyframes and text queries."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """Compute CLIP embeddings for images and text.

    Uses open_clip to load a pretrained CLIP model and produce normalised
    embeddings for both visual frames and text queries.

    Args:
        model_name: OpenCLIP model name.
        pretrained: Pretrained weights identifier.
        device: Torch device string.
    """

    EMBEDDING_DIM = 512  # ViT-B-32 default

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._device = device
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None

    def connect(self) -> None:
        """Load the CLIP model, preprocess transform, and tokenizer."""
        model, _, preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained, device=self._device,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self._model_name)
        logger.info("CLIP model '%s' loaded on %s", self._model_name, self._device)

    def embed_image(self, frame: np.ndarray) -> list[float]:
        """Compute a normalised CLIP embedding for a single BGR frame.

        Args:
            frame: NumPy array of shape (H, W, 3) in BGR colour order.

        Returns:
            List of floats — the normalised CLIP embedding.
        """
        if self._model is None:
            raise RuntimeError("Call connect() before embed_image()")

        rgb = frame[:, :, ::-1]  # BGR → RGB
        pil_img = Image.fromarray(rgb)
        tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features /= features.norm(dim=-1, keepdim=True)

        return features.squeeze().cpu().tolist()

    def embed_images_batch(self, frames: list[np.ndarray]) -> list[list[float]]:
        """Compute CLIP embeddings for a batch of frames."""
        if self._model is None:
            raise RuntimeError("Call connect() before embed_images_batch()")

        tensors = []
        for frame in frames:
            rgb = frame[:, :, ::-1]
            pil_img = Image.fromarray(rgb)
            tensors.append(self._preprocess(pil_img))

        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(batch)
            features /= features.norm(dim=-1, keepdim=True)

        return features.cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        """Compute a normalised CLIP text embedding for a query string."""
        if self._model is None:
            raise RuntimeError("Call connect() before embed_text()")

        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)

        return features.squeeze().cpu().tolist()
