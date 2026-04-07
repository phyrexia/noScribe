# MeetingGenie - Editor Page
# Plain text editor for transcripts and summaries with open/save/search

import os
import flet as ft
from app_state import AppState
from event_bus import EventType

BRAND_BLUE = "#0A84FF"


def build_editor_page(page: ft.Page, state: AppState) -> ft.Control:

    current_file = {"path": None, "modified": False}

    # ---- File picker ----
    file_picker = ft.FilePicker()
    save_picker = ft.FilePicker()
    page.services.extend([file_picker, save_picker])

    # ---- Editor area ----
    editor = ft.TextField(
        multiline=True,
        min_lines=30,
        expand=True,
        text_size=14,
        border=ft.InputBorder.NONE,
        content_padding=ft.padding.all(16),
    )

    file_label = ft.Text("No file open", size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)

    # ---- Functions ----
    def _load_file(path: str):
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                editor.value = f.read()
            current_file["path"] = path
            current_file["modified"] = False
            file_label.value = os.path.basename(path)
            file_label.italic = False
            editor.update()
            file_label.update()
        except Exception as e:
            file_label.value = f"Error: {e}"
            file_label.update()

    def _save_file(path: str = None):
        save_path = path or current_file["path"]
        if not save_path:
            return
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(editor.value or "")
            current_file["path"] = save_path
            current_file["modified"] = False
            file_label.value = os.path.basename(save_path)
            file_label.italic = False
            file_label.update()
        except Exception as e:
            file_label.value = f"Save error: {e}"
            file_label.update()

    # ---- Open/Save handlers ----
    async def _on_open_result():
        result = await file_picker.pick_files(
            dialog_title="Open file",
            allowed_extensions=["txt", "srt", "vtt", "html", "md"],
        )
        if result:
            _load_file(result[0].path)

    def on_open(e):
        page.run_task(_on_open_result)

    async def _on_save_as_result():
        result = await save_picker.save_file(
            dialog_title="Save as",
            file_name=os.path.basename(current_file["path"]) if current_file["path"] else "document.txt",
            allowed_extensions=["txt", "srt", "vtt", "md"],
        )
        if result:
            _save_file(result)

    def on_save(e):
        if current_file["path"]:
            _save_file()
        else:
            page.run_task(_on_save_as_result)

    def on_save_as(e):
        page.run_task(_on_save_as_result)

    # ---- Search ----
    search_field = ft.TextField(
        label="Search",
        width=250,
        dense=True,
        visible=False,
        on_submit=lambda e: _find_next(),
    )
    replace_field = ft.TextField(
        label="Replace",
        width=200,
        dense=True,
        visible=False,
    )
    search_visible = {"on": False}

    def _toggle_search(e):
        search_visible["on"] = not search_visible["on"]
        search_field.visible = search_visible["on"]
        replace_field.visible = search_visible["on"]
        search_row.update()
        if search_visible["on"]:
            search_field.focus()

    def _find_next():
        if not search_field.value or not editor.value:
            return
        text = editor.value
        query = search_field.value
        # Find from current cursor or start
        pos = text.find(query)
        if pos >= 0:
            editor.selection = ft.TextSelection(
                base_offset=pos,
                extent_offset=pos + len(query),
            )
            editor.update()

    def _replace_all(e):
        if not search_field.value or not editor.value:
            return
        editor.value = editor.value.replace(search_field.value, replace_field.value or "")
        editor.update()

    # ---- Zoom ----
    font_size = {"current": 14}

    def _zoom_in(e):
        font_size["current"] = min(font_size["current"] + 2, 32)
        editor.text_size = font_size["current"]
        editor.update()

    def _zoom_out(e):
        font_size["current"] = max(font_size["current"] - 2, 8)
        editor.text_size = font_size["current"]
        editor.update()

    # ---- Listen for events (auto-open after transcription/summary) ----
    def _on_job_finished(payload):
        summary_path = payload.get("summary_path")
        transcript_path = payload.get("transcript_path")
        # Prefer summary, fallback to transcript
        target = summary_path or transcript_path
        if target and os.path.exists(target):
            _load_file(target)

    state.bus.subscribe(EventType.JOB_FINISHED, _on_job_finished)

    # ---- Toolbar ----
    toolbar = ft.Row(
        [
            ft.IconButton(icon=ft.Icons.FOLDER_OPEN, tooltip="Open", on_click=on_open),
            ft.IconButton(icon=ft.Icons.SAVE, tooltip="Save (Ctrl+S)", on_click=on_save),
            ft.IconButton(icon=ft.Icons.SAVE_AS, tooltip="Save As", on_click=on_save_as),
            ft.VerticalDivider(width=1),
            ft.IconButton(icon=ft.Icons.SEARCH, tooltip="Search & Replace", on_click=_toggle_search),
            ft.VerticalDivider(width=1),
            ft.IconButton(icon=ft.Icons.ZOOM_IN, tooltip="Zoom in", on_click=_zoom_in),
            ft.IconButton(icon=ft.Icons.ZOOM_OUT, tooltip="Zoom out", on_click=_zoom_out),
            ft.Container(expand=True),
            file_label,
        ],
        spacing=4,
    )

    search_row = ft.Row(
        [
            search_field,
            ft.IconButton(icon=ft.Icons.NAVIGATE_NEXT, tooltip="Find next", on_click=lambda e: _find_next(), icon_size=18),
            replace_field,
            ft.TextButton("Replace All", on_click=_replace_all),
        ],
        spacing=4,
        visible=True,
    )

    # ---- Layout ----
    return ft.Container(
        content=ft.Column(
            [
                ft.Text("Editor", size=20, weight=ft.FontWeight.BOLD),
                toolbar,
                search_row,
                ft.Container(
                    content=editor,
                    expand=True,
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                    border_radius=8,
                ),
            ],
            spacing=8,
            expand=True,
        ),
        padding=20,
        expand=True,
    )
