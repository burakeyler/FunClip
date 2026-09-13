import copy
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "funclip"))

from videoclipper import VideoClipper  # noqa: E402


CUE_TIMES = re.compile(r"(\d\d:\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)")


class TestAudioClipSubtitleTimeline(unittest.TestCase):
    """Subtitles of concatenated audio regions follow the concatenated output."""

    def setUp(self):
        sentences = [
            {"text": "hello", "timestamp": [[1000, 2000]], "spk": 0},
            {"text": "world", "timestamp": [[4000, 5000]], "spk": 0},
        ]
        self.state = {
            "audio_input": (16000, np.zeros(96000)),
            "recog_res_raw": "hello world",
            "timestamp": [[1000, 2000], [4000, 5000]],
            "sentences": sentences,
            "sd_sentences": copy.deepcopy(sentences),
        }

    def clip(self, *args, **kwargs):
        return VideoClipper(None).clip(*args, state=copy.deepcopy(self.state), **kwargs)

    def test_second_region_subtitle_starts_after_the_first_region(self):
        (rate, audio), _, subtitles = self.clip("hello#world", 0, 0)

        self.assertEqual((rate, len(audio)), (16000, 32000))
        self.assertEqual(
            CUE_TIMES.findall(subtitles),
            [("00:00:00,000", "00:00:01,000"), ("00:00:01,000", "00:00:02,000")],
        )

    def test_explicit_timestamps_follow_the_concatenated_output(self):
        (_, audio), _, subtitles = self.clip(
            None, 0, 0, timestamp_list=[[16000, 32000], [64000, 80000]]
        )

        self.assertEqual(len(audio), 32000)
        self.assertEqual(
            CUE_TIMES.findall(subtitles),
            [("00:00:00,000", "00:00:01,000"), ("00:00:01,000", "00:00:02,000")],
        )

    def test_single_region_is_unchanged(self):
        (_, audio), _, subtitles = self.clip("hello", 0, 0)

        self.assertEqual(len(audio), 16000)
        self.assertEqual(CUE_TIMES.findall(subtitles), [("00:00:00,000", "00:00:01,000")])


if __name__ == "__main__":
    unittest.main()
