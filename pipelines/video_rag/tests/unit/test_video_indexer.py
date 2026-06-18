"""Unit tests for VideoIndexer — mock Whisper and subprocess calls."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video_rag.video_indexer import TranscriptSegment, VideoIndexer, VideoMetadata


class TestTranscriptSegment:
    def test_frozen_dataclass(self) -> None:
        seg = TranscriptSegment(start_ts=0.0, end_ts=1.5, text="Hello world")
        assert seg.start_ts == 0.0
        assert seg.end_ts == 1.5
        assert seg.text == "Hello world"

    def test_immutable(self) -> None:
        seg = TranscriptSegment(start_ts=0.0, end_ts=1.0, text="test")
        with pytest.raises(AttributeError):
            seg.start_ts = 2.0  # type: ignore[misc]


class TestVideoIndexer:
    def test_init_defaults(self) -> None:
        indexer = VideoIndexer()
        assert indexer._whisper_model_name == "base"
        assert indexer._device == "cpu"

    def test_init_custom_params(self) -> None:
        indexer = VideoIndexer(whisper_model="large", device="cuda")
        assert indexer._whisper_model_name == "large"
        assert indexer._device == "cuda"

    @patch("video_rag.video_indexer.whisper")
    def test_connect_loads_model(self, mock_whisper: MagicMock) -> None:
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model

        indexer = VideoIndexer(whisper_model="small")
        indexer.connect()

        mock_whisper.load_model.assert_called_once_with("small", device="cpu")
        assert indexer._model is mock_model

    @patch("video_rag.video_indexer.whisper")
    def test_transcribe_returns_segments(self, mock_whisper: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Hello everyone"},
                {"start": 2.5, "end": 5.0, "text": "Welcome to this lecture"},
                {"start": 5.0, "end": 8.0, "text": "Today we discuss AI"},
            ]
        }
        mock_whisper.load_model.return_value = mock_model

        indexer = VideoIndexer()
        indexer.connect()
        segments = indexer.transcribe(Path("/tmp/test.wav"))

        assert len(segments) == 3
        assert segments[0].start_ts == 0.0
        assert segments[0].end_ts == 2.5
        assert segments[0].text == "Hello everyone"
        assert segments[2].text == "Today we discuss AI"

    @patch("video_rag.video_indexer.whisper")
    def test_transcribe_skips_empty_segments(self, mock_whisper: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Real content"},
                {"start": 1.0, "end": 2.0, "text": "   "},
                {"start": 2.0, "end": 3.0, "text": ""},
                {"start": 3.0, "end": 4.0, "text": "More content"},
            ]
        }
        mock_whisper.load_model.return_value = mock_model

        indexer = VideoIndexer()
        indexer.connect()
        segments = indexer.transcribe(Path("/tmp/test.wav"))

        assert len(segments) == 2
        assert segments[0].text == "Real content"
        assert segments[1].text == "More content"

    def test_transcribe_raises_without_connect(self) -> None:
        indexer = VideoIndexer()
        with pytest.raises(RuntimeError, match="Call connect"):
            indexer.transcribe(Path("/tmp/test.wav"))

    @patch("subprocess.run")
    @patch("video_rag.video_indexer.whisper")
    def test_extract_audio(self, mock_whisper: MagicMock, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)

        indexer = VideoIndexer()
        result = indexer.extract_audio(Path("/tmp/video.mp4"))

        assert result == Path("/tmp/video.wav")
        mock_run.assert_called_once()


class TestVideoMetadata:
    def test_metadata_fields(self) -> None:
        meta = VideoMetadata(
            video_id="v1",
            title="Test Video",
            url="https://youtube.com/watch?v=123",
            duration_s=120.0,
            segment_count=10,
            file_path="/tmp/v1.mp4",
            topics=["ai", "ml"],
        )
        assert meta.video_id == "v1"
        assert meta.segment_count == 10
        assert meta.topics == ["ai", "ml"]

    def test_metadata_default_topics(self) -> None:
        meta = VideoMetadata(
            video_id="v1", title="T", url="u", duration_s=0, segment_count=0, file_path="p"
        )
        assert meta.topics == []
