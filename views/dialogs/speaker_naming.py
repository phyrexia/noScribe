# MeetingGenie - Speaker Naming Dialog
# Uses page.pubsub to communicate between worker thread and UI thread.

import os
import platform
import shlex
import threading
from subprocess import Popen, DEVNULL
import flet as ft

BRAND_BLUE = "#0A84FF"

_play_proc = None

def _ms_to_ts(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS."""
    s = ms // 1000
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _play_audio_segment(audio_path: str, start_ms: int, end_ms: int, app_dir: str):
    """Play an audio segment using ffmpeg to extract + afplay to play."""
    global _play_proc
    # Kill any existing playback
    if _play_proc and _play_proc.poll() is None:
        try:
            _play_proc.terminate()
        except Exception:
            pass

    duration_ms = end_ms - start_ms
    if duration_ms <= 0:
        print(f"[speaker play] invalid duration: {start_ms}-{end_ms}")
        return

    if not os.path.exists(audio_path):
        print(f"[speaker play] audio not found: {audio_path}")
        return

    # Find ffmpeg
    ffmpeg_arm64 = os.path.join(app_dir, 'ffmpeg-arm64')
    ffmpeg_std = os.path.join(app_dir, 'ffmpeg')
    if platform.machine() == "arm64" and os.path.exists(ffmpeg_arm64):
        ffmpeg = ffmpeg_arm64
    elif os.path.exists(ffmpeg_std):
        ffmpeg = ffmpeg_std
    else:
        ffmpeg = 'ffmpeg'  # system fallback

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()

    # Extract segment
    cmd = f'"{ffmpeg}" -loglevel warning -y -ss {start_ms}ms -t {duration_ms}ms -i "{audio_path}" -ar 44100 -ac 1 "{tmp.name}"'
    if platform.system() != "Windows":
        cmd = shlex.split(cmd)

    try:
        p = Popen(cmd, stdout=DEVNULL, stderr=DEVNULL)
        rc = p.wait(timeout=10)
        if rc != 0:
            print(f"[speaker play] ffmpeg failed with code {rc}")
            return

        # Check file has content
        if os.path.getsize(tmp.name) < 100:
            print(f"[speaker play] extracted file too small: {os.path.getsize(tmp.name)} bytes")
            return

        if platform.system() == "Darwin":
            _play_proc = Popen(['afplay', tmp.name])
        else:
            _play_proc = Popen(['ffplay', '-nodisp', '-autoexit', tmp.name], stdout=DEVNULL, stderr=DEVNULL)
        print(f"[speaker play] playing {_ms_to_ts(start_ms)}-{_ms_to_ts(end_ms)}")
    except Exception as e:
        print(f"[speaker play] error: {e}")


class SpeakerNamingBridge:
    """Bridge between worker thread and UI thread for speaker naming dialog.

    Usage:
      1. Call bridge.setup(page) once during page build (UI thread)
      2. From worker thread: result = bridge.request_naming(speakers_data, audio_path)
    """

    def __init__(self):
        self._done_event = threading.Event()
        self._result = {}
        self._page = None
        self._dlg = None

    def setup(self, page: ft.Page):
        """Call from UI thread during page setup."""
        self._page = page

    def request_naming(self, speakers_data: list, audio_path: str, app_dir: str = '') -> dict:
        """Call from worker thread. Blocks until user responds. Returns {label: name}."""
        self._done_event.clear()
        self._result = {}
        self._audio_path = audio_path
        self._app_dir = app_dir

        # Build dialog controls
        name_fields = {}
        save_checkboxes = {}
        rows = []

        for spk in speakers_data:
            lbl = spk['label']
            short = spk['short_label']
            matched = spk.get('matched_name', '')
            sim = spk.get('similarity', 0.0)

            default_name = matched if matched and sim > 0.7 else short

            name_field = ft.TextField(
                value=default_name,
                label=short,
                width=200,
                dense=True,
            )
            name_fields[lbl] = name_field

            if matched and sim > 0:
                pct = int(sim * 100)
                badge_color = "#4CAF50" if sim > 0.8 else "#FF9800" if sim > 0.6 else "#9E9E9E"
                confidence = ft.Container(
                    content=ft.Text(f"{matched} ({pct}%)", size=11, color=ft.Colors.WHITE),
                    bgcolor=badge_color,
                    border_radius=10,
                    padding=ft.padding.symmetric(horizontal=8, vertical=2),
                )
            else:
                confidence = ft.Container(width=0)

            save_cb = ft.Checkbox(
                label="Save voice",
                value=bool(matched and sim > 0.7),
            )
            save_checkboxes[lbl] = save_cb

            # Audio sample play buttons with timestamps
            samples = spk.get('samples', [])
            play_buttons = []
            for idx, sample in enumerate(samples[:2]):
                s_start = sample.get('start', 0)
                s_end = sample.get('end', 0)
                dur = round((s_end - s_start) / 1000, 1)
                ts_label = f"{_ms_to_ts(s_start)}"

                def _make_play(st=s_start, en=s_end):
                    def _play(e):
                        threading.Thread(
                            target=_play_audio_segment,
                            args=(self._audio_path, st, en, self._app_dir),
                            daemon=True,
                        ).start()
                    return _play

                play_buttons.append(
                    ft.TextButton(
                        content=ft.Row([
                            ft.Icon(ft.Icons.PLAY_ARROW, size=16),
                            ft.Text(f"{ts_label} ({dur}s)", size=11),
                        ], spacing=2, tight=True),
                        tooltip=f"Play sample at {ts_label}",
                        on_click=_make_play(),
                    )
                )

            # Build row: name | play buttons | confidence | save
            row_controls = [name_field]
            if play_buttons:
                row_controls.append(ft.Row(play_buttons, spacing=2))
            row_controls.append(confidence)
            row_controls.append(save_cb)
            rows.append(ft.Row(row_controls, spacing=6))

        def _on_ok(e):
            result = {}
            for lbl, field in name_fields.items():
                name = field.value.strip()
                if name:
                    result[lbl] = name
                    if save_checkboxes[lbl].value:
                        spk_data = next((s for s in speakers_data if s['label'] == lbl), None)
                        if spk_data and spk_data.get('embedding'):
                            try:
                                import speaker_db
                                speaker_db.save_speaker(name, spk_data['embedding'])
                            except Exception:
                                pass
            self._result = result
            self._page.pop_dialog()
            self._done_event.set()

        def _on_skip(e):
            self._result = {}
            self._page.pop_dialog()
            self._done_event.set()

        self._dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Identify Speakers"),
            content=ft.Column(
                [
                    ft.Text(
                        "Assign names to detected speakers.\n"
                        "Check 'Save voice' to remember for future transcriptions.",
                        size=13,
                    ),
                    ft.Divider(height=8),
                    *rows,
                ],
                tight=True,
                spacing=10,
                width=450,
                scroll=ft.ScrollMode.AUTO,
            ),
            actions=[
                ft.TextButton("Skip", on_click=_on_skip),
                ft.ElevatedButton("OK", on_click=_on_ok,
                                  bgcolor=BRAND_BLUE, color=ft.Colors.WHITE),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        # Use pubsub to trigger dialog open on UI thread
        self._page.pubsub.send_all("__show_speaker_dialog__")

        # Block worker thread
        self._done_event.wait(timeout=300)
        return self._result

    def on_pubsub_message(self, message):
        """Subscribe this to page.pubsub. Called on UI thread."""
        if message == "__show_speaker_dialog__" and self._dlg:
            self._page.show_dialog(self._dlg)
