# MeetingGenie - Settings Page

import flet as ft
from app_state import AppState
from config import get_config, set_config, save_config

BRAND_BLUE = "#0A84FF"

DEFAULT_SUMMARY_PROMPT = (
    "You are an expert executive assistant and meeting scribe. "
    "Your task is to analyze the following meeting transcript and provide a highly structured, "
    "clear, and professional summary. Include:\n"
    "1. Executive Summary (2-3 sentences max)\n"
    "2. Key Discussion Points (Bullet points)\n"
    "3. Decisions Made (If any)\n"
    "4. Action Items (Assignee and Task, if identifiable)\n\n"
    "Only output the summary, nothing else. Provide the output in the same language as the transcript."
)


def build_settings_page(page: ft.Page, state: AppState) -> ft.Control:

    def _save_all(e=None):
        set_config('anthropic_api_key', api_key_field.value.strip())
        set_config('summary_prompt', prompt_field.value.strip())
        set_config('summary_model', model_field.value.strip())
        set_config('summary_max_tokens', int(max_tokens_field.value or 1500))
        set_config('summary_temperature', float(temp_field.value or 0.3))
        set_config('proxy_url', proxy_field.value.strip())
        set_config('ignore_ssl', 'true' if ssl_cb.value else 'false')
        set_config('whisper_beam_size', int(beam_field.value or 1))
        set_config('auto_save', 'True' if autosave_cb.value else 'False')
        save_config()
        page.open(ft.SnackBar(ft.Text("Settings saved.")))

    def _reset_prompt(e):
        prompt_field.value = DEFAULT_SUMMARY_PROMPT
        prompt_field.update()

    # ---- AI Summary section ----
    api_key_field = ft.TextField(
        label="Anthropic API Key",
        value=get_config('anthropic_api_key', ''),
        password=True,
        can_reveal_password=True,
        width=400,
    )

    model_field = ft.TextField(
        label="Summary model",
        value=get_config('summary_model', 'claude-sonnet-4-20250514'),
        width=400,
        hint_text="e.g. claude-sonnet-4-20250514",
    )

    prompt_field = ft.TextField(
        label="Summary system prompt",
        value=get_config('summary_prompt', DEFAULT_SUMMARY_PROMPT),
        multiline=True,
        min_lines=4,
        max_lines=8,
        width=500,
    )

    max_tokens_field = ft.TextField(
        label="Max tokens",
        value=str(get_config('summary_max_tokens', 1500)),
        width=120,
        input_filter=ft.InputFilter(r"[0-9]"),
    )

    temp_field = ft.TextField(
        label="Temperature",
        value=str(get_config('summary_temperature', 0.3)),
        width=120,
    )

    # ---- Transcription section ----
    beam_field = ft.TextField(
        label="Whisper beam size",
        value=str(get_config('whisper_beam_size', 1)),
        width=120,
        input_filter=ft.InputFilter(r"[0-9]"),
    )

    autosave_cb = ft.Checkbox(
        label="Auto-save transcript during processing",
        value=get_config('auto_save', 'True') != 'False',
    )

    # ---- Network section ----
    proxy_field = ft.TextField(
        label="Proxy URL",
        value=get_config('proxy_url', ''),
        width=400,
        hint_text="http://proxy.corp.com:8080 (leave blank for system proxy)",
    )

    ssl_cb = ft.Checkbox(
        label="Bypass SSL verification (corporate proxies)",
        value=get_config('ignore_ssl', 'false').lower() == 'true',
    )

    # ---- Speaker DB section ----
    speaker_table = ft.Column(spacing=4)

    def _build_speaker_row(name, info):
        """Build a row for one speaker with rename, delete actions."""
        name_field = ft.TextField(value=name, dense=True, width=160, read_only=True)
        created = info.get("created", "?") if info else "?"
        updated = info.get("updated", "?") if info else "?"

        def _on_rename(e):
            if name_field.read_only:
                name_field.read_only = False
                name_field.focus()
                name_field.update()
            else:
                new_name = name_field.value.strip()
                if new_name and new_name != name:
                    import speaker_db
                    speaker_db.rename_speaker(name, new_name)
                    page.open(ft.SnackBar(ft.Text(f"Renamed: {name} → {new_name}")))
                    _list_speakers()
                else:
                    name_field.read_only = True
                    name_field.update()

        def _on_delete(e):
            import speaker_db
            speaker_db.delete_speaker(name)
            page.open(ft.SnackBar(ft.Text(f"Deleted: {name}")))
            _list_speakers()

        def _on_merge(e):
            """Show merge dialog to pick which speaker to merge into this one."""
            import speaker_db
            others = [n for n in speaker_db.list_speakers() if n.lower() != name.lower()]
            if not others:
                page.open(ft.SnackBar(ft.Text("No other speakers to merge with.")))
                return

            merge_dropdown = ft.Dropdown(
                label="Merge into this speaker",
                options=[ft.dropdown.Option(n) for n in others],
                width=200,
            )

            def _do_merge(e):
                target = merge_dropdown.value
                if target:
                    sim = speaker_db.get_similarity(name, target)
                    speaker_db.merge_speakers(name, target)
                    page.close(merge_dlg)
                    page.open(ft.SnackBar(
                        ft.Text(f"Merged {target} into {name} (similarity: {int(sim*100)}%)")
                    ))
                    _list_speakers()

            merge_dlg = ft.AlertDialog(
                modal=True,
                title=ft.Text(f"Merge into '{name}'"),
                content=ft.Column([
                    ft.Text("Select speaker to absorb. Their voice signature will be blended.", size=13),
                    merge_dropdown,
                ], tight=True, width=300),
                actions=[
                    ft.TextButton("Cancel", on_click=lambda e: page.close(merge_dlg)),
                    ft.ElevatedButton("Merge", on_click=_do_merge, bgcolor="#FF9800", color=ft.Colors.WHITE),
                ],
            )
            page.open(merge_dlg)

        return ft.Container(
            content=ft.Row([
                name_field,
                ft.Text(f"Created: {created}", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text(f"Updated: {updated}", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.IconButton(icon=ft.Icons.EDIT, tooltip="Rename", on_click=_on_rename, icon_size=18),
                ft.IconButton(icon=ft.Icons.MERGE, tooltip="Merge with another speaker", on_click=_on_merge, icon_size=18),
                ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="Delete", on_click=_on_delete, icon_size=18, icon_color="#FF453A"),
            ], spacing=8, alignment=ft.MainAxisAlignment.START),
            border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
            border_radius=8,
            padding=ft.padding.symmetric(horizontal=12, vertical=6),
        )

    def _list_speakers(e=None):
        try:
            import speaker_db
            names = speaker_db.list_speakers()
            if names:
                speaker_table.controls = []
                for n in names:
                    info = speaker_db.get_speaker_info(n)
                    speaker_table.controls.append(_build_speaker_row(n, info))
            else:
                speaker_table.controls = [ft.Text("No saved speakers.", italic=True, size=13)]
            speaker_table.update()
        except Exception as ex:
            speaker_table.controls = [ft.Text(f"Error: {ex}", size=13, color="#FF453A")]
            speaker_table.update()

    # ---- Layout ----
    save_btn = ft.ElevatedButton(
        "Save Settings",
        icon=ft.Icons.SAVE,
        bgcolor=BRAND_BLUE,
        color=ft.Colors.WHITE,
        on_click=_save_all,
        width=200,
    )

    content = ft.Column(
        [
            ft.Text("Settings", size=24, weight=ft.FontWeight.BOLD),
            ft.Divider(height=16),

            # AI Summary
            ft.Text("AI Summary", size=18, weight=ft.FontWeight.W_600),
            ft.Text("Configure the Anthropic Claude API for meeting summaries.", size=13,
                     color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Divider(height=8, color=ft.Colors.TRANSPARENT),
            api_key_field,
            model_field,
            ft.Row([max_tokens_field, temp_field], spacing=16),
            prompt_field,
            ft.TextButton("Reset prompt to default", on_click=_reset_prompt, icon=ft.Icons.RESTORE),
            ft.Divider(height=16),

            # Transcription
            ft.Text("Transcription", size=18, weight=ft.FontWeight.W_600),
            ft.Divider(height=8, color=ft.Colors.TRANSPARENT),
            beam_field,
            autosave_cb,
            ft.Divider(height=16),

            # Network
            ft.Text("Network", size=18, weight=ft.FontWeight.W_600),
            ft.Divider(height=8, color=ft.Colors.TRANSPARENT),
            proxy_field,
            ssl_cb,
            ft.Divider(height=16),

            # Speaker DB
            ft.Text("Saved Speakers", size=18, weight=ft.FontWeight.W_600),
            ft.Text("Voice signatures saved for automatic speaker recognition.", size=13,
                     color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Divider(height=8, color=ft.Colors.TRANSPARENT),
            ft.TextButton("Refresh speaker list", on_click=_list_speakers, icon=ft.Icons.REFRESH),
            speaker_table,
            ft.Divider(height=24),

            save_btn,
        ],
        spacing=8,
        scroll=ft.ScrollMode.AUTO,
        expand=True,
    )

    # Load speakers on first view
    _list_speakers()

    return ft.Container(content=content, padding=20, expand=True)
