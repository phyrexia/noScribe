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
    s = ms // 1000
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _play_audio_segment(audio_path: str, start_ms: int, end_ms: int, app_dir: str):
    """Play an audio segment using ffmpeg to extract + afplay to play."""
    global _play_proc
    if _play_proc and _play_proc.poll() is None:
        try:
            _play_proc.terminate()
        except Exception:
            pass

    duration_ms = end_ms - start_ms
    if duration_ms <= 0:
        return
    if not os.path.exists(audio_path):
        print(f"[speaker play] audio not found: {audio_path}")
        return

    ffmpeg_arm64 = os.path.join(app_dir, 'ffmpeg-arm64')
    ffmpeg_std = os.path.join(app_dir, 'ffmpeg')
    if platform.machine() == "arm64" and os.path.exists(ffmpeg_arm64):
        ffmpeg = ffmpeg_arm64
    elif os.path.exists(ffmpeg_std):
        ffmpeg = ffmpeg_std
    else:
        ffmpeg = 'ffmpeg'

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()

    cmd = f'"{ffmpeg}" -loglevel warning -y -ss {start_ms}ms -t {duration_ms}ms -i "{audio_path}" -ar 44100 -ac 1 "{tmp.name}"'
    if platform.system() != "Windows":
        cmd = shlex.split(cmd)

    try:
        p = Popen(cmd, stdout=DEVNULL, stderr=DEVNULL)
        rc = p.wait(timeout=10)
        if rc != 0:
            return
        if os.path.getsize(tmp.name) < 100:
            return
        if platform.system() == "Darwin":
            _play_proc = Popen(['afplay', tmp.name])
        else:
            _play_proc = Popen(['ffplay', '-nodisp', '-autoexit', tmp.name], stdout=DEVNULL, stderr=DEVNULL)
    except Exception as e:
        print(f"[speaker play] error: {e}")


class SpeakerNamingBridge:
    def __init__(self):
        self._done_event = threading.Event()
        self._result = {}
        self._page = None
        self._dlg = None

    def setup(self, page: ft.Page):
        self._page = page

    def request_naming(self, speakers_data: list, audio_path: str, app_dir: str = '') -> dict:
        """Call from worker thread. Blocks until user responds (up to 2h).
        Returns {label: name} dict."""
        self._done_event.clear()
        self._result = {}
        self._audio_path = audio_path
        self._app_dir = app_dir
        self._saved_names = []

        # Deduplicate speakers by label
        seen_labels = set()
        unique_speakers = []
        for spk in speakers_data:
            if spk['label'] not in seen_labels:
                seen_labels.add(spk['label'])
                unique_speakers.append(spk)

        name_fields = {}
        save_checkboxes = {}
        rows = []

        for spk in unique_speakers:
            lbl = spk['label']
            short = spk['short_label']
            matched = spk.get('matched_name', '')
            sim = spk.get('similarity', 0.0)

            # Pre-fill: matched name if good confidence, else short label
            default_name = matched if matched and sim > 0.7 else short
            # Auto-check save if it's a new speaker (no match)
            auto_save = bool(matched and sim > 0.7)

            name_field = ft.TextField(
                value=default_name,
                label=short,
                width=180,
                dense=True,
            )
            name_fields[lbl] = name_field

            # Confidence badge
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

            save_cb = ft.Checkbox(label="Save", value=auto_save, tooltip="Save voice signature")
            save_checkboxes[lbl] = save_cb

            # Audio sample play buttons
            samples = spk.get('samples', [])
            play_buttons = []
            for idx, sample in enumerate(samples[:2]):
                s_start = sample.get('start', 0)
                s_end = sample.get('end', 0)
                dur = round((s_end - s_start) / 1000, 1)
                ts_label = _ms_to_ts(s_start)

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
                        on_click=_make_play(),
                    )
                )

            # Compact card: top row = name + save, bottom row = play + confidence
            top_row = ft.Row([name_field, save_cb], spacing=8)
            bottom_items = []
            if play_buttons:
                bottom_items.extend(play_buttons)
            if matched and sim > 0:
                bottom_items.append(confidence)
            bottom_row = ft.Row(bottom_items, spacing=4, wrap=True) if bottom_items else ft.Container(height=0)

            rows.append(
                ft.Container(
                    content=ft.Column([top_row, bottom_row], spacing=4, tight=True),
                    padding=ft.padding.symmetric(horizontal=10, vertical=6),
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                    border_radius=8,
                )
            )

        # Status text for feedback
        status_text = ft.Text("", size=12, color="#4CAF50")

        def _on_ok(e):
            result = {}
            saved = []
            for lbl, field in name_fields.items():
                name = field.value.strip()
                if name:
                    result[lbl] = name
                    if save_checkboxes[lbl].value:
                        spk_data = next((s for s in unique_speakers if s['label'] == lbl), None)
                        if spk_data and spk_data.get('embedding'):
                            try:
                                import speaker_db
                                speaker_db.save_speaker(name, spk_data['embedding'])
                                saved.append(name)
                            except Exception:
                                pass
            self._saved_names = saved
            self._result = result
            self._page.pop_dialog()
            self._done_event.set()

        def _on_skip(e):
            self._result = {}
            self._saved_names = []
            self._page.pop_dialog()
            self._done_event.set()

        n_speakers = len(unique_speakers)
        self._dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Identify {n_speakers} Speakers"),
            content=ft.Column(
                [
                    ft.Text(
                        "Assign names to each speaker. Use the play buttons to hear samples.\n"
                        "Check 'Save' to remember voice signatures for future transcriptions.\n"
                        "Take your time — transcription will wait for you.",
                        size=13,
                    ),
                    ft.Divider(height=8),
                    *rows,
                    status_text,
                ],
                tight=True,
                spacing=10,
                width=420,
                scroll=ft.ScrollMode.AUTO,
            ),
            actions=[
                ft.TextButton("Skip (use S01, S02...)", on_click=_on_skip),
                ft.ElevatedButton("OK — Start Transcription", on_click=_on_ok,
                                  bgcolor=BRAND_BLUE, color=ft.Colors.WHITE),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        # Trigger dialog on UI thread
        self._page.pubsub.send_all("__show_speaker_dialog__")

        # Block worker thread — 2 hour timeout
        self._done_event.wait(timeout=7200)

        # Log what was saved
        if self._saved_names:
            print(f"[speakers] Saved voice signatures: {', '.join(self._saved_names)}")

        return self._result

    def on_pubsub_message(self, message):
        if message == "__show_speaker_dialog__" and self._dlg:
            self._page.show_dialog(self._dlg)
