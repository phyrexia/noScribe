# MeetingGenie - Settings Page

import os
import platform
from subprocess import Popen, DEVNULL

import flet as ft
from app_state import AppState
from config import get_config, set_config, save_config

BRAND_BLUE = "#0A84FF"

# How many sample-play buttons to render per signature row.
MAX_SAMPLES_IN_UI = 4

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
        set_config('summary_model', model_field.value or 'claude-sonnet-4-20250514')
        set_config('summary_max_tokens', int(max_tokens_field.value or 1500))
        set_config('summary_temperature', float(temp_field.value or 0.3))
        set_config('proxy_url', proxy_field.value.strip())
        set_config('ignore_ssl', 'true' if ssl_cb.value else 'false')
        set_config('whisper_beam_size', int(beam_field.value or 1))
        set_config('auto_save', 'True' if autosave_cb.value else 'False')
        set_config('diarization_backend', diar_backend_field.value or 'pyannote')
        save_config()
        page.show_dialog(ft.SnackBar(ft.Text("Settings saved.")))

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

    model_field = ft.Dropdown(
        label="Summary model",
        value=get_config('summary_model', 'claude-sonnet-4-20250514'),
        width=400,
        options=[
            ft.dropdown.Option("claude-sonnet-4-20250514", "Claude Sonnet 4 (fast, cheap)"),
            ft.dropdown.Option("claude-opus-4-20250514", "Claude Opus 4 (most capable)"),
            ft.dropdown.Option("claude-haiku-4-20250414", "Claude Haiku 4 (fastest, cheapest)"),
            ft.dropdown.Option("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet (legacy)"),
        ],
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

    diar_backend_field = ft.Dropdown(
        label="Diarization backend",
        value=get_config('diarization_backend', 'pyannote'),
        width=400,
        options=[
            ft.dropdown.Option("pyannote", "pyannote 3.1 (default, PyTorch/MPS)"),
            ft.dropdown.Option(
                "sherpa-onnx",
                "sherpa-onnx (ONNX + CoreML/CPU, experimental)",
            ),
        ],
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

    # ---- Voice Signature Manager ----
    speaker_table = ft.Column(spacing=4)
    similar_panel = ft.Column(spacing=4)
    sort_state = {"by": "last_used"}  # 'name' | 'last_used' | 'use_count'

    def _play_sample(path: str):
        """Play a stored sample WAV via `afplay` (macOS) or `ffplay` fallback."""
        import speaker_db
        abs_path = speaker_db.sample_abs_path(path)
        if not os.path.exists(abs_path):
            page.show_dialog(ft.SnackBar(ft.Text(f"Sample missing: {os.path.basename(abs_path)}")))
            return
        try:
            if platform.system() == "Darwin":
                Popen(["afplay", abs_path])
            else:
                Popen(["ffplay", "-nodisp", "-autoexit", abs_path], stdout=DEVNULL, stderr=DEVNULL)
        except Exception as ex:
            page.show_dialog(ft.SnackBar(ft.Text(f"Play error: {ex}")))

    def _relative_time(iso_str: str) -> str:
        if not iso_str or iso_str == '?':
            return '—'
        try:
            from datetime import datetime as _dt
            # Tolerate trailing Z
            s = iso_str.rstrip('Z')
            try:
                dt = _dt.fromisoformat(s)
            except ValueError:
                # Date-only legacy value
                dt = _dt.fromisoformat(s + 'T00:00:00')
            delta = _dt.utcnow() - dt
            secs = int(delta.total_seconds())
            if secs < 0:
                return 'just now'
            if secs < 60:
                return f"{secs}s ago"
            if secs < 3600:
                return f"{secs // 60}m ago"
            if secs < 86400:
                return f"{secs // 3600}h ago"
            days = secs // 86400
            if days < 30:
                return f"{days}d ago"
            if days < 365:
                return f"{days // 30}mo ago"
            return f"{days // 365}y ago"
        except Exception:
            return iso_str

    def _build_speaker_row(spk: dict):
        """Render one signature row: rename, samples, stats, delete."""
        sid = spk.get('id') or ''
        name = spk.get('name', '')
        last_used = spk.get('last_used', '')
        use_count = int(spk.get('use_count', 0) or 0)
        created = spk.get('created', '?')
        samples = list(spk.get('samples', []) or [])

        name_field = ft.TextField(
            value=name, dense=True, width=180,
            tooltip="Rename — saves on blur",
        )

        def _on_blur(e):
            import speaker_db
            new_name = (name_field.value or '').strip()
            if new_name and new_name != name:
                speaker_db.rename_speaker(sid or name, new_name)
                page.show_dialog(ft.SnackBar(ft.Text(f"Renamed: {name} → {new_name}")))
                _refresh()
        name_field.on_blur = _on_blur

        def _on_delete(e):
            def _confirm(e2):
                import speaker_db
                speaker_db.delete_speaker(sid or name)
                page.pop_dialog()
                page.show_dialog(ft.SnackBar(ft.Text(f"Deleted: {name}")))
                _refresh()

            confirm_dlg = ft.AlertDialog(
                modal=True,
                title=ft.Text(f"Delete '{name}'?"),
                content=ft.Text(
                    "The voice signature and its stored WAV samples will be permanently removed.",
                    size=13,
                ),
                actions=[
                    ft.TextButton("Cancel", on_click=lambda e: page.pop_dialog()),
                    ft.ElevatedButton(
                        "Delete", on_click=_confirm,
                        bgcolor="#FF453A", color=ft.Colors.WHITE,
                    ),
                ],
            )
            page.show_dialog(confirm_dlg)

        # Audio play buttons for stored samples (compact)
        play_buttons = []
        for idx, sp in enumerate(samples[:MAX_SAMPLES_IN_UI], start=1):
            def _make_play(p=sp):
                return lambda e: _play_sample(p)
            play_buttons.append(
                ft.IconButton(
                    icon=ft.Icons.PLAY_CIRCLE_OUTLINE,
                    tooltip=f"Play sample {idx}",
                    icon_size=18,
                    on_click=_make_play(),
                )
            )
        if not play_buttons:
            play_buttons.append(
                ft.Text("no samples", size=10, italic=True,
                        color=ft.Colors.ON_SURFACE_VARIANT)
            )

        meta = ft.Row(
            [
                ft.Text(f"used {use_count}×", size=11,
                        color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text(f"last {_relative_time(last_used)}", size=11,
                        color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text(f"created {created}", size=11,
                        color=ft.Colors.ON_SURFACE_VARIANT),
            ],
            spacing=10,
        )

        return ft.Container(
            content=ft.Row(
                [
                    name_field,
                    ft.Row(play_buttons, spacing=2, tight=True),
                    meta,
                    ft.Container(expand=True),
                    ft.IconButton(
                        icon=ft.Icons.DELETE_OUTLINE,
                        tooltip="Delete signature",
                        on_click=_on_delete,
                        icon_size=18,
                        icon_color="#FF453A",
                    ),
                ],
                spacing=8,
                alignment=ft.MainAxisAlignment.START,
            ),
            border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
            border_radius=8,
            padding=ft.padding.symmetric(horizontal=12, vertical=6),
        )

    def _sort_speakers(speakers):
        by = sort_state["by"]
        if by == "name":
            return sorted(speakers, key=lambda s: (s.get('name') or '').lower())
        if by == "use_count":
            return sorted(speakers, key=lambda s: int(s.get('use_count', 0) or 0),
                          reverse=True)
        # last_used (default)
        return sorted(speakers, key=lambda s: s.get('last_used') or '',
                      reverse=True)

    def _build_similar_panel():
        """List up to 10 most-suspicious near-duplicate pairs."""
        import speaker_db
        speakers = {s['id']: s for s in speaker_db.list_speakers_full() if s.get('id')}
        rows = []
        pairs = [
            p for p in speaker_db.pairwise_similarity()
            if 0.6 <= p[2] < 1.0 and p[0] in speakers and p[1] in speakers
        ]
        if not pairs:
            return [
                ft.Text("No suspicious near-duplicates (≥ 60 %).", size=12, italic=True,
                        color=ft.Colors.ON_SURFACE_VARIANT),
            ]
        for ida, idb, sim in pairs[:10]:
            a = speakers[ida]
            b = speakers[idb]
            pct = int(sim * 100)
            badge_color = "#FF453A" if sim > 0.85 else "#FF9800" if sim > 0.7 else "#9E9E9E"

            def _make_merge(keep, drop, keep_name, drop_name):
                def _do_merge(e):
                    import speaker_db as sd
                    surviving = sd.merge_speakers(keep, drop)
                    if surviving:
                        page.show_dialog(ft.SnackBar(
                            ft.Text(f"Merged '{drop_name}' into '{keep_name}'"),
                        ))
                    _refresh()
                return _do_merge

            rows.append(
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Container(
                                content=ft.Text(f"{pct}%", size=11, weight=ft.FontWeight.BOLD,
                                                color=ft.Colors.WHITE),
                                bgcolor=badge_color,
                                padding=ft.padding.symmetric(horizontal=8, vertical=2),
                                border_radius=4,
                            ),
                            ft.Text(a['name'], size=13, weight=ft.FontWeight.W_500),
                            ft.Text("↔", size=13, color=ft.Colors.ON_SURFACE_VARIANT),
                            ft.Text(b['name'], size=13, weight=ft.FontWeight.W_500),
                            ft.Container(expand=True),
                            ft.TextButton(
                                f"Merge into {a['name']}",
                                icon=ft.Icons.MERGE,
                                on_click=_make_merge(ida, idb, a['name'], b['name']),
                            ),
                            ft.TextButton(
                                f"Merge into {b['name']}",
                                icon=ft.Icons.MERGE,
                                on_click=_make_merge(idb, ida, b['name'], a['name']),
                            ),
                        ],
                        spacing=8,
                    ),
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                    border_radius=8,
                    padding=ft.padding.symmetric(horizontal=10, vertical=4),
                )
            )
        return rows

    def _refresh(e=None):
        try:
            import speaker_db
            speakers = speaker_db.list_speakers_full()
            if speakers:
                speaker_table.controls = [
                    _build_speaker_row(s) for s in _sort_speakers(speakers)
                ]
            else:
                speaker_table.controls = [ft.Text("No saved speakers.", italic=True, size=13)]
            similar_panel.controls = _build_similar_panel()
            try:
                speaker_table.update()
                similar_panel.update()
            except Exception:
                pass
        except Exception as ex:
            speaker_table.controls = [ft.Text(f"Error: {ex}", size=13, color="#FF453A")]
            try:
                speaker_table.update()
            except Exception:
                pass

    def _on_sort_change(e):
        sort_state["by"] = e.control.value or "last_used"
        _refresh()

    sort_dropdown = ft.Dropdown(
        label="Sort by",
        value="last_used",
        options=[
            ft.dropdown.Option("last_used", "Last used"),
            ft.dropdown.Option("name", "Name (A→Z)"),
            ft.dropdown.Option("use_count", "Usage count"),
        ],
        on_select=_on_sort_change,
        width=180,
        dense=True,
    )

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
            diar_backend_field,
            ft.Divider(height=16),

            # Network
            ft.Text("Network", size=18, weight=ft.FontWeight.W_600),
            ft.Divider(height=8, color=ft.Colors.TRANSPARENT),
            proxy_field,
            ssl_cb,
            ft.Divider(height=16),

            # Voice Signature Manager
            ft.Text("Voice Signatures", size=18, weight=ft.FontWeight.W_600),
            ft.Text(
                "Per-speaker voice models used for automatic recognition. "
                "Rename, play stored samples, merge duplicates or delete entries you no longer need.",
                size=13, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            ft.Divider(height=8, color=ft.Colors.TRANSPARENT),
            ft.Row([
                sort_dropdown,
                ft.TextButton("Refresh", on_click=_refresh, icon=ft.Icons.REFRESH),
            ], spacing=12),
            speaker_table,
            ft.Divider(height=16, color=ft.Colors.TRANSPARENT),
            ft.Text("Suspicious near-duplicates", size=14, weight=ft.FontWeight.W_600),
            ft.Text(
                "Pairs with cosine similarity between 60 % and 100 %. "
                "Merge them if they actually refer to the same person.",
                size=12, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            similar_panel,
            ft.Divider(height=24),

            save_btn,
        ],
        spacing=8,
        scroll=ft.ScrollMode.AUTO,
        expand=True,
    )

    # Pre-populate signature list (without .update() since not yet on page)
    try:
        import speaker_db
        speakers = speaker_db.list_speakers_full()
        if speakers:
            for s in _sort_speakers(speakers):
                speaker_table.controls.append(_build_speaker_row(s))
        else:
            speaker_table.controls.append(ft.Text("No saved speakers.", italic=True, size=13))
        similar_panel.controls.extend(_build_similar_panel())
    except Exception:
        speaker_table.controls.append(ft.Text("Error loading speakers.", size=13, color="#FF453A"))

    return ft.Container(content=content, padding=20, expand=True)
