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

            default_name = matched if matched and sim > 0.7 else short
            auto_save = bool(matched and sim > 0.7)

            name_field = ft.TextField(value=default_name, label=short, width=220, dense=True)
            name_fields[lbl] = name_field

            save_cb = ft.Checkbox(value=auto_save, tooltip="Save voice")
            save_checkboxes[lbl] = save_cb

            # Match badge (compact)
            if matched and sim > 0:
                pct = int(sim * 100)
                badge_color = "#4CAF50" if sim > 0.8 else "#FF9800" if sim > 0.6 else "#9E9E9E"
                badge = ft.Text(f"{pct}%", size=10, color=ft.Colors.WHITE,
                                bgcolor=badge_color, weight=ft.FontWeight.BOLD)
            else:
                badge = ft.Container(width=0)

            # Play buttons (icon only, compact)
            samples = spk.get('samples', [])
            play_btns = []
            for idx, sample in enumerate(samples[:2]):
                s_start = sample.get('start', 0)
                s_end = sample.get('end', 0)
                dur = round((s_end - s_start) / 1000, 1)

                def _make_play(st=s_start, en=s_end):
                    def _play(e):
                        threading.Thread(
                            target=_play_audio_segment,
                            args=(self._audio_path, st, en, self._app_dir),
                            daemon=True,
                        ).start()
                    return _play

                play_btns.append(
                    ft.TextButton(
                        content=ft.Row([
                            ft.Icon(ft.Icons.PLAY_ARROW, size=14),
                            ft.Text(f"{_ms_to_ts(s_start)}", size=10),
                        ], spacing=1, tight=True),
                        tooltip=f"Play {dur}s sample",
                        on_click=_make_play(),
                        style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=4, vertical=2)),
                    )
                )

            # Single row: [Name] [▶][▶] [98%] [☑]
            row_items = [name_field]
            row_items.extend(play_btns)
            row_items.append(badge)
            row_items.append(save_cb)
            rows.append(ft.Row(row_items, spacing=2))

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
                    ft.Text("▶ = play sample  |  % = voice match  |  ☑ = save signature", size=11,
                            color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Divider(height=4),
                    *rows,
                ],
                tight=True,
                spacing=6,
                width=480,
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
