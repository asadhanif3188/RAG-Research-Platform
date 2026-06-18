"""VideoIndexer — transcribe video audio with Whisper and segment into timestamped chunks."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import whisper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptSegment:
    """A single transcript segment with precise timestamps."""

    start_ts: float
    end_ts: float
    text: str


class VideoIndexer:
    """Download (if URL) and transcribe videos using OpenAI Whisper.

    Produces a list of :class:`TranscriptSegment` with word-level timestamps
    aggregated into sentence-sized chunks.

    Args:
        whisper_model: Whisper model size (tiny, base, small, medium, large).
        device: Torch device string (cpu, cuda).
        download_dir: Directory for yt-dlp downloads.
    """

    def __init__(
        self,
        whisper_model: str = "base",
        device: str = "cpu",
        download_dir: str | None = None,
    ) -> None:
        self._whisper_model_name = whisper_model
        self._device = device
        self._download_dir = download_dir or tempfile.gettempdir()
        self._model: Any = None

    def connect(self) -> None:
        """Load the Whisper model into memory."""
        self._model = whisper.load_model(self._whisper_model_name, device=self._device)
        logger.info(
            "Whisper model '%s' loaded on %s", self._whisper_model_name, self._device
        )

    def download_video(self, url: str) -> Path:
        """Download a video from a URL using yt-dlp and return the local path."""
        output_template = str(Path(self._download_dir) / "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "--format", "bestaudio[ext=m4a]/bestaudio/best",
            "--output", output_template,
            "--no-playlist",
            "--quiet",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

        # Find the downloaded file
        cmd_path = [
            "yt-dlp",
            "--get-filename",
            "--format", "bestaudio[ext=m4a]/bestaudio/best",
            "--output", output_template,
            "--no-playlist",
            url,
        ]
        path_result = subprocess.run(cmd_path, capture_output=True, text=True, check=True)
        downloaded = Path(path_result.stdout.strip())
        logger.info("Downloaded video to %s", downloaded)
        return downloaded

    def extract_audio(self, video_path: Path) -> Path:
        """Extract audio track from a video file to WAV format."""
        audio_path = video_path.with_suffix(".wav")
        if audio_path.exists():
            return audio_path

        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(audio_path),
            "-y", "-loglevel", "error",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Extracted audio to %s", audio_path)
        return audio_path

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        """Transcribe an audio file and return timestamped segments.

        Each segment corresponds roughly to a sentence boundary detected by Whisper.
        """
        if self._model is None:
            raise RuntimeError("Call connect() before transcribe()")

        result = self._model.transcribe(
            str(audio_path),
            word_timestamps=True,
            verbose=False,
        )

        segments: list[TranscriptSegment] = []
        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    start_ts=round(seg["start"], 2),
                    end_ts=round(seg["end"], 2),
                    text=text,
                )
            )

        logger.info(
            "Transcribed %d segments from %s (%.1fs total)",
            len(segments),
            audio_path.name,
            segments[-1].end_ts if segments else 0.0,
        )
        return segments

    def index_video(self, source: str) -> tuple[Path, list[TranscriptSegment]]:
        """Full pipeline: download (if URL) → extract audio → transcribe.

        Args:
            source: Local file path or YouTube/web URL.

        Returns:
            Tuple of (video_path, transcript_segments).
        """
        video_path = Path(source)
        if source.startswith(("http://", "https://")):
            video_path = self.download_video(source)

        audio_path = self.extract_audio(video_path)
        segments = self.transcribe(audio_path)
        return video_path, segments


@dataclass
class VideoMetadata:
    """Metadata about an indexed video."""

    video_id: str
    title: str
    url: str
    duration_s: float
    segment_count: int
    file_path: str
    topics: list[str] = field(default_factory=list)
