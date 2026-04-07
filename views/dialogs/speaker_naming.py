# MeetingGenie - Speaker Naming Dialog
# Shows detected speakers and lets user assign names

import threading
import flet as ft

BRAND_BLUE = "#0A84FF"


def show_speaker_naming_dialog(page: ft.Page, speakers_data: list, audio_path: str) -> dict:
    """Show a dialog for naming speakers. Blocks until user responds.

    Called from a worker thread — uses threading.Event to block.
    Returns {label: name} dict, or {} if skipped.
    """
    result_holder = [{}]
    done_event = threading.Event()

    name_fields = {}
    save_checkboxes = {}

    rows = []
    for spk in speakers_data:
        lbl = spk['label']
        short = spk['short_label']
        matched = spk.get('matched_name', '')
        sim = spk.get('similarity', 0.0)

        # Pre-fill with matched name if confidence > 0.7
        default_name = matched if matched and sim > 0.7 else short

        name_field = ft.TextField(
            value=default_name,
            label=short,
            width=200,
            dense=True,
        )
        name_fields[lbl] = name_field

        # Confidence badge
        if matched and sim > 0:
            pct = int(sim * 100)
            if sim > 0.8:
                badge_color = "#4CAF50"
            elif sim > 0.6:
                badge_color = "#FF9800"
            else:
                badge_color = "#9E9E9E"
            confidence = ft.Container(
                content=ft.Text(f"{matched} ({pct}%)", size=11, color=ft.Colors.WHITE),
                bgcolor=badge_color,
                border_radius=10,
                padding=ft.padding.symmetric(horizontal=8, vertical=2),
            )
        else:
            confidence = ft.Container(width=0)

        save_cb = ft.Checkbox(
            label="Save",
            value=bool(matched and sim > 0.7),
            tooltip="Save voice signature for future recognition",
        )
        save_checkboxes[lbl] = save_cb

        rows.append(
            ft.Row([name_field, confidence, save_cb], spacing=8)
        )

    def on_ok(e):
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
        result_holder[0] = result
        page.close(dlg)
        done_event.set()

    def on_skip(e):
        result_holder[0] = {}
        page.close(dlg)
        done_event.set()

    dlg = ft.AlertDialog(
        modal=True,
        title=ft.Text("Identify Speakers"),
        content=ft.Column(
            [
                ft.Text(
                    "Assign names to detected speakers.\nCheck 'Save' to remember voice signatures for future use.",
                    size=13,
                ),
                ft.Divider(height=8),
                *rows,
            ],
            tight=True,
            spacing=10,
            width=450,
        ),
        actions=[
            ft.TextButton("Skip", on_click=on_skip),
            ft.ElevatedButton("OK", on_click=on_ok, bgcolor=BRAND_BLUE, color=ft.Colors.WHITE),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    # Open dialog from UI thread
    page.open(dlg)
    page.update()

    # Block worker thread until user responds
    done_event.wait(timeout=300)
    return result_holder[0]
