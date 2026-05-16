"""Smoke tests for the PyAV-backed audio converter."""

from __future__ import annotations

import math
import os
import struct
import sys
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

# Allow `python -m unittest` from the repo root and direct invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audio.convert import (  # noqa: E402
    AudioConversionCanceled,
    ToWav,
    convert_to_wav,
)


def _write_stereo_wav(path: str, seconds: float = 1.0, rate: int = 44100) -> None:
    n = int(seconds * rate)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            sample = int(math.sin(2 * math.pi * 440 * (i / rate)) * 0.3 * 32767)
            frames += struct.pack("<hh", sample, sample)
        w.writeframes(bytes(frames))


class ConvertToWavTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = os.path.join(self.tmp.name, "src.wav")
        self.dst = os.path.join(self.tmp.name, "dst.wav")
        _write_stereo_wav(self.src, seconds=1.0, rate=44100)

    def test_full_conversion_produces_16k_mono_pcm_s16le(self) -> None:
        convert_to_wav(self.src, self.dst)
        with wave.open(self.dst, "rb") as r:
            self.assertEqual(r.getframerate(), 16000)
            self.assertEqual(r.getnchannels(), 1)
            self.assertEqual(r.getsampwidth(), 2)
            # ~1s of audio at 16 kHz → ~16000 frames (allow small codec slack).
            self.assertGreater(r.getnframes(), 15000)
            self.assertLess(r.getnframes(), 17000)

    def test_partial_window_is_shorter_than_full(self) -> None:
        full = os.path.join(self.tmp.name, "full.wav")
        part = os.path.join(self.tmp.name, "part.wav")
        convert_to_wav(self.src, full)
        convert_to_wav(self.src, part, start_ms=200, stop_ms=600)
        with wave.open(full, "rb") as r:
            full_frames = r.getnframes()
        with wave.open(part, "rb") as r:
            part_frames = r.getnframes()
            self.assertEqual(r.getframerate(), 16000)
            self.assertEqual(r.getnchannels(), 1)
        self.assertLess(part_frames, full_frames)
        # ~400 ms window @16 kHz = 6400 frames, allow generous slack for
        # encoder padding / seek granularity.
        self.assertGreater(part_frames, 4000)
        self.assertLess(part_frames, 9000)

    def test_cancel_raises(self) -> None:
        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] > 1  # cancel after first frame

        with self.assertRaises(AudioConversionCanceled):
            with ToWav(self.src, self.dst, should_cancel=cancel) as conv:
                conv.run()


if __name__ == "__main__":
    unittest.main()
