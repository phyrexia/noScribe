# MeetingGenie - Speaker Naming Dialog
# Uses page.pubsub to communicate between worker thread and UI thread.

import threading
import flet as ft

BRAND_BLUE = "#0A84FF"


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

    def request_naming(self, speakers_data: list, audio_path: str) -> dict:
        """Call from worker thread. Blocks until user responds. Returns {label: name}."""
        self._done_event.clear()
        self._result = {}

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
            rows.append(ft.Row([name_field, confidence, save_cb], spacing=8))

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
