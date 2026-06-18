"""Unit tests for CLIPEmbedder — mock CLIP model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from video_rag.clip_embedder import CLIPEmbedder


class TestCLIPEmbedder:
    def test_init_defaults(self) -> None:
        embedder = CLIPEmbedder()
        assert embedder._model_name == "ViT-B-32"
        assert embedder._device == "cpu"

    def test_embed_image_raises_without_connect(self) -> None:
        embedder = CLIPEmbedder()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Call connect"):
            embedder.embed_image(frame)

    def test_embed_text_raises_without_connect(self) -> None:
        embedder = CLIPEmbedder()
        with pytest.raises(RuntimeError, match="Call connect"):
            embedder.embed_text("hello")

    @patch("video_rag.clip_embedder.open_clip")
    def test_connect_loads_model(self, mock_open_clip: MagicMock) -> None:
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model,
            None,
            mock_preprocess,
        )
        mock_open_clip.get_tokenizer.return_value = MagicMock()

        embedder = CLIPEmbedder(model_name="ViT-B-32", pretrained="test")
        embedder.connect()

        mock_open_clip.create_model_and_transforms.assert_called_once()
        assert embedder._model is mock_model
        assert embedder._preprocess is mock_preprocess

    def test_embed_image_returns_list(self) -> None:
        """Test embed_image with manually set mock model."""
        embedder = CLIPEmbedder()

        fake_embedding = torch.randn(1, 512)
        fake_embedding = fake_embedding / fake_embedding.norm(dim=-1, keepdim=True)

        mock_model = MagicMock()
        mock_model.encode_image.return_value = fake_embedding

        mock_tensor = MagicMock()
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor

        mock_preprocess = MagicMock(return_value=mock_tensor)

        embedder._model = mock_model
        embedder._preprocess = mock_preprocess

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = embedder.embed_image(frame)

        assert isinstance(result, list)
        assert len(result) == 512

    def test_embed_text_returns_list(self) -> None:
        """Test embed_text with manually set mock model."""
        embedder = CLIPEmbedder()

        fake_embedding = torch.randn(1, 512)
        fake_embedding = fake_embedding / fake_embedding.norm(dim=-1, keepdim=True)

        mock_model = MagicMock()
        mock_model.encode_text.return_value = fake_embedding

        mock_tokenizer = MagicMock()
        tokens_mock = MagicMock()
        tokens_mock.to.return_value = tokens_mock
        mock_tokenizer.return_value = tokens_mock

        embedder._model = mock_model
        embedder._tokenizer = mock_tokenizer

        result = embedder.embed_text("a cat sitting on a mat")

        assert isinstance(result, list)
        assert len(result) == 512
