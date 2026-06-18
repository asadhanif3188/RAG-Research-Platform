"""Unit tests for SceneDetector — use synthetic video frames."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from video_rag.scene_detector import Keyframe, SceneDetector


class TestKeyframe:
    def test_frozen_dataclass(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        kf = Keyframe(timestamp_s=1.5, frame_index=45, frame=frame)
        assert kf.timestamp_s == 1.5
        assert kf.frame_index == 45
        assert kf.frame.shape == (100, 100, 3)


class TestSceneDetector:
    def test_init_defaults(self) -> None:
        detector = SceneDetector()
        assert detector._threshold == 0.4
        assert detector._min_scene_duration_s == 2.0
        assert detector._sample_fps == 2.0

    def test_init_custom(self) -> None:
        detector = SceneDetector(threshold=0.3, min_scene_duration_s=1.0, sample_fps=5.0)
        assert detector._threshold == 0.3
        assert detector._min_scene_duration_s == 1.0
        assert detector._sample_fps == 5.0

    @patch("video_rag.scene_detector.cv2")
    def test_detect_scenes_returns_keyframes(self, mock_cv2: MagicMock) -> None:
        """Simulate a video with distinct scenes."""
        bright = np.full((100, 100, 3), 200, dtype=np.uint8)
        dark = np.full((100, 100, 3), 20, dtype=np.uint8)

        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.get.return_value = 30.0

        frames = [bright] * 30 + [dark] * 30
        call_count = 0

        def read_side_effect():
            nonlocal call_count
            if call_count < len(frames):
                frame = frames[call_count]
                call_count += 1
                return True, frame
            return False, None

        cap_mock.read.side_effect = read_side_effect
        mock_cv2.VideoCapture.return_value = cap_mock
        mock_cv2.COLOR_BGR2GRAY = 6
        mock_cv2.HISTCMP_BHATTACHARYYA = 3

        mock_cv2.cvtColor.side_effect = lambda frame, code: frame[:, :, 0]

        def mock_calc_hist(images, channels, mask, hist_size, ranges):
            img = images[0]
            hist = np.histogram(img.flatten(), bins=64, range=(0, 256))[0]
            return hist.reshape(-1, 1).astype(np.float32)

        mock_cv2.calcHist.side_effect = mock_calc_hist
        mock_cv2.normalize.side_effect = lambda h, h2: h

        def mock_compare_hist(h1, h2, method):
            diff = float(np.abs(h1.mean() - h2.mean()))
            return min(diff / 10.0, 1.0)

        mock_cv2.compareHist.side_effect = mock_compare_hist

        detector = SceneDetector(threshold=0.3, min_scene_duration_s=0.5, sample_fps=2.0)
        keyframes = detector.detect_scenes(Path("/tmp/test.mp4"))

        assert len(keyframes) >= 1
        assert keyframes[0].timestamp_s == 0.0

    @patch("video_rag.scene_detector.cv2")
    def test_detect_scenes_empty_video(self, mock_cv2: MagicMock) -> None:
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.get.return_value = 30.0
        cap_mock.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = cap_mock

        detector = SceneDetector()
        keyframes = detector.detect_scenes(Path("/tmp/empty.mp4"))
        assert keyframes == []

    @patch("video_rag.scene_detector.cv2")
    def test_detect_scenes_unopenable_video(self, mock_cv2: MagicMock) -> None:
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = cap_mock

        detector = SceneDetector()
        with pytest.raises(RuntimeError, match="Cannot open video"):
            detector.detect_scenes(Path("/tmp/bad.mp4"))
