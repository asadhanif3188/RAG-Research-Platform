"""SceneDetector — extract keyframes at scene boundaries using frame difference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Keyframe:
    """A single keyframe extracted at a scene change."""

    timestamp_s: float
    frame_index: int
    frame: np.ndarray  # BGR image (H, W, 3)


class SceneDetector:
    """Detect scene changes in a video and extract one keyframe per scene.

    Uses histogram difference between consecutive frames to detect shot boundaries.
    When the difference exceeds the threshold, a new scene is registered and the
    frame is captured.

    Args:
        threshold: Histogram difference threshold for scene change detection (0–1).
            Lower values = more sensitive. Default 0.4 works well for lectures.
        min_scene_duration_s: Minimum seconds between scene changes to avoid
            rapid-fire detections.
        sample_fps: Frames per second to sample (down from source FPS).
    """

    def __init__(
        self,
        threshold: float = 0.4,
        min_scene_duration_s: float = 2.0,
        sample_fps: float = 2.0,
    ) -> None:
        self._threshold = threshold
        self._min_scene_duration_s = min_scene_duration_s
        self._sample_fps = sample_fps

    def detect_scenes(self, video_path: Path) -> list[Keyframe]:
        """Detect scene changes and extract keyframes.

        Args:
            video_path: Path to the video file.

        Returns:
            List of Keyframe objects, one per detected scene.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(source_fps / self._sample_fps))
        min_frame_gap = int(self._min_scene_duration_s * source_fps)

        keyframes: list[Keyframe] = []
        prev_hist: np.ndarray | None = None
        frame_idx = 0
        last_scene_frame = -min_frame_gap  # allow first frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
                cv2.normalize(hist, hist)

                if prev_hist is not None:
                    diff = cv2.compareHist(
                        prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA
                    )
                    if diff > self._threshold and (frame_idx - last_scene_frame) >= min_frame_gap:
                        timestamp = frame_idx / source_fps
                        keyframes.append(
                            Keyframe(
                                timestamp_s=round(timestamp, 2),
                                frame_index=frame_idx,
                                frame=frame.copy(),
                            )
                        )
                        last_scene_frame = frame_idx
                else:
                    # Always capture the first frame as a keyframe
                    keyframes.append(
                        Keyframe(
                            timestamp_s=0.0,
                            frame_index=0,
                            frame=frame.copy(),
                        )
                    )
                    last_scene_frame = frame_idx

                prev_hist = hist

            frame_idx += 1

        cap.release()

        logger.info(
            "Detected %d scenes in %s (%.1fs video)",
            len(keyframes),
            video_path.name,
            frame_idx / source_fps if source_fps else 0,
        )
        return keyframes
