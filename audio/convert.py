"""Decode arbitrary media to 16 kHz mono PCM s16le WAV using PyAV.

The output is byte-equivalent to running:

    ffmpeg -ss <start_ms>ms -i <input> [-t <duration_ms>ms] \
           -ar 16000 -ac 1 -c:a pcm_s16le <output>

which is the canonical input shape expected by the downstream pyannote +
faster-whisper pipeline.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import av
from av.audio.resampler import AudioResampler


TARGET_RATE = 16000
TARGET_LAYOUT = "mono"
TARGET_FORMAT = "s16"           # interleaved 16-bit signed PCM
TARGET_CODEC = "pcm_s16le"
TARGET_CONTAINER = "wav"


class AudioConversionError(RuntimeError):
    """Raised when the underlying decoder/encoder fails."""


class AudioConversionCanceled(RuntimeError):
    """Raised when the user-supplied cancel callback returns True."""


class ToWav:
    """Context-managed converter.

    Typical usage::

        with ToWav(in_path, out_path, start_ms=0, stop_ms=60_000,
                   should_cancel=lambda: app.cancel) as conv:
            conv.run()
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        start_ms: Optional[int] = None,
        stop_ms: Optional[int] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.input_path = os.fspath(input_path)
        self.output_path = os.fspath(output_path)
        self.start_ms = int(start_ms) if start_ms else 0
        self.stop_ms = int(stop_ms) if stop_ms else 0
        self.should_cancel = should_cancel or (lambda: False)

        self._input: Optional[av.container.InputContainer] = None
        self._output: Optional[av.container.OutputContainer] = None
        self._in_stream = None
        self._out_stream = None
        self._resampler: Optional[AudioResampler] = None
        self._closed = False

    # ------------------------------------------------------------------ context

    def __enter__(self) -> "ToWav":
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------ open

    def _open(self) -> None:
        self._input = av.open(self.input_path)
        audio_streams = [s for s in self._input.streams if s.type == "audio"]
        if not audio_streams:
            raise AudioConversionError(f"No audio stream found in {self.input_path}")
        self._in_stream = audio_streams[0]
        # Limit decoding work to the audio stream we care about.
        for s in self._input.streams:
            s.thread_type = "AUTO" if s is self._in_stream else "NONE"

        self._output = av.open(self.output_path, mode="w", format=TARGET_CONTAINER)
        self._out_stream = self._output.add_stream(TARGET_CODEC, rate=TARGET_RATE)
        # PyAV >= 14 prefers AudioLayout objects but accepts the string alias.
        try:
            self._out_stream.layout = TARGET_LAYOUT
        except Exception:
            pass
        try:
            self._out_stream.channels = 1
        except Exception:
            pass
        self._out_stream.format = TARGET_FORMAT

        self._resampler = AudioResampler(
            format=TARGET_FORMAT,
            layout=TARGET_LAYOUT,
            rate=TARGET_RATE,
        )

        if self.start_ms > 0:
            # av.seek wants stream time-base units; use the container default
            # (AV_TIME_BASE = 1e6) so the math is platform-independent.
            seek_ts = int(self.start_ms * 1000)  # ms -> microseconds
            try:
                self._input.seek(seek_ts, any_frame=False, backward=True)
            except av.AVError as e:
                raise AudioConversionError(f"Seek to {self.start_ms} ms failed: {e}") from e

    # ------------------------------------------------------------------ run

    def run(self) -> None:
        if self._input is None or self._output is None:
            raise AudioConversionError("Converter is not open")

        start_s = self.start_ms / 1000.0
        stop_s = (self.stop_ms / 1000.0) if self.stop_ms > 0 else None

        try:
            for frame in self._input.decode(self._in_stream):
                if self.should_cancel():
                    raise AudioConversionCanceled()

                # Drop pre-roll frames the demuxer handed us before start.
                pts = frame.pts
                if pts is not None and frame.time_base is not None:
                    frame_time = float(pts * frame.time_base)
                    if frame_time + (frame.samples / float(frame.sample_rate or TARGET_RATE)) <= start_s:
                        continue
                    if stop_s is not None and frame_time >= stop_s:
                        break

                # Resamplers in modern PyAV return a list of frames.
                for resampled in self._resampler.resample(frame):
                    if resampled is None:
                        continue
                    for packet in self._out_stream.encode(resampled):
                        self._output.mux(packet)

            # Flush resampler and encoder.
            for resampled in self._resampler.resample(None):
                if resampled is None:
                    continue
                for packet in self._out_stream.encode(resampled):
                    self._output.mux(packet)
            for packet in self._out_stream.encode(None):
                self._output.mux(packet)
        except AudioConversionCanceled:
            raise
        except av.AVError as e:
            raise AudioConversionError(str(e)) from e

    # ------------------------------------------------------------------ close

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._output is not None:
                self._output.close()
        finally:
            self._output = None
            try:
                if self._input is not None:
                    self._input.close()
            finally:
                self._input = None
                self._resampler = None
                self._in_stream = None
                self._out_stream = None


def convert_to_wav(
    input_path: str,
    output_path: str,
    start_ms: Optional[int] = None,
    stop_ms: Optional[int] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Functional shortcut around :class:`ToWav`."""
    with ToWav(
        input_path,
        output_path,
        start_ms=start_ms,
        stop_ms=stop_ms,
        should_cancel=should_cancel,
    ) as conv:
        conv.run()
