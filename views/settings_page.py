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
    def _list_speakers(e=None):
        try:
            import speaker_db
            names = speaker_db.list_speakers()
            if names:
                speaker_list.controls = [
                    ft.Chip(ft.Text(n), on_delete=lambda e, name=n: _delete_speaker(name))
                    for n in names
                ]
            else:
                speaker_list.controls = [ft.Text("No saved speakers.", italic=True, size=13)]
            speaker_list.update()
        except Exception:
            speaker_list.controls = [ft.Text("Error loading speakers.", size=13, color="#FF453A")]
            speaker_list.update()

    def _delete_speaker(name):
        try:
            import speaker_db
            speaker_db.delete_speaker(name)
            _list_speakers()
            page.open(ft.SnackBar(ft.Text(f"Deleted speaker: {name}")))
        except Exception as ex:
            page.open(ft.SnackBar(ft.Text(f"Error: {ex}")))

    speaker_list = ft.Row(wrap=True, spacing=8)

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
            speaker_list,
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
