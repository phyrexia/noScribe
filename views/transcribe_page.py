# MeetingGenie - Transcribe Page
# File selection, options, start button, log output, progress bar

import os
import threading
import flet as ft
from app_state import AppState
from event_bus import EventType
from models import TranscriptionJob, create_transcription_job
from config import get_config
import model_manager

BRAND_BLUE = "#0A84FF"


def build_transcribe_page(page: ft.Page, state: AppState) -> ft.Control:
    """Build the Transcribe page and return the root control."""

    # ---- File display labels ------------------------------------------
    audio_file_text = ft.Text("No file selected", size=13, italic=True, expand=True)
    transcript_file_text = ft.Text("Auto", size=13, italic=True, expand=True)

    # ---- Async file pick handlers -------------------------------------
    audio_picker = ft.FilePicker()
    transcript_picker = ft.FilePicker()
    page.services.extend([audio_picker, transcript_picker])

    async def _do_pick_audio():
        result = await audio_picker.pick_files(
            dialog_title="Select audio file",
            allowed_extensions=["wav", "mp3", "m4a", "ogg", "flac", "wma", "aac", "mp4", "mkv", "webm", "mov", "avi", "wmv", "m4v", "mpg", "mpeg", "3gp"],
            allow_multiple=True,
        )
        if result:
            paths = [f.path for f in result]
            state.audio_files = paths
            names = ", ".join(os.path.basename(p) for p in paths)
            audio_file_text.value = names
            audio_file_text.italic = False
            if paths:
                base = os.path.splitext(paths[0])[0]
                ext = filetype_dropdown.value or "txt"
                state.transcript_files = [f"{base}.{ext}"]
                transcript_file_text.value = os.path.basename(state.transcript_files[0])
                transcript_file_text.italic = False
            page.update()

    def pick_audio(e):
        page.run_task(_do_pick_audio)

    async def _do_pick_transcript():
        result = await transcript_picker.save_file(
            dialog_title="Set transcript file",
            file_name="transcript.txt",
            allowed_extensions=["txt", "html", "srt", "vtt"],
        )
        if result:
            state.transcript_files = [result]
            transcript_file_text.value = os.path.basename(result)
            transcript_file_text.italic = False
            page.update()

    def pick_transcript(e):
        page.run_task(_do_pick_transcript)

    # ---- Options section ----------------------------------------------
    language_dropdown = ft.Dropdown(
        label="Language",
        value="Auto",
        options=[ft.dropdown.Option(name) for name in state.get_language_names()],
        width=220,
        dense=True,
    )

    model_dropdown = ft.Dropdown(
        label="Model",
        value="precise",
        options=[
            ft.dropdown.Option("fast", "Fast (310 MB)"),
            ft.dropdown.Option("precise", "Precise (1.5 GB)"),
        ],
        width=220,
        dense=True,
    )

    speaker_dropdown = ft.Dropdown(
        label="Speaker detection",
        value="auto",
        options=[
            ft.dropdown.Option("auto"),
            ft.dropdown.Option("off"),
            ft.dropdown.Option("1"),
            ft.dropdown.Option("2"),
            ft.dropdown.Option("3"),
            ft.dropdown.Option("4"),
            ft.dropdown.Option("5"),
        ],
        width=220,
        dense=True,
    )

    pause_dropdown = ft.Dropdown(
        label="Mark pauses",
        value="1sec+",
        options=[
            ft.dropdown.Option("none"),
            ft.dropdown.Option("1sec+"),
            ft.dropdown.Option("2sec+"),
            ft.dropdown.Option("3sec+"),
        ],
        width=220,
        dense=True,
    )

    filetype_dropdown = ft.Dropdown(
        label="Output format",
        value="txt",
        options=[
            ft.dropdown.Option("txt", "Plain text (.txt)"),
            ft.dropdown.Option("html", "HTML (.html)"),
            ft.dropdown.Option("srt", "Subtitles (.srt)"),
            ft.dropdown.Option("vtt", "WebVTT (.vtt)"),
        ],
        width=220,
        dense=True,
    )

    overlapping_cb = ft.Checkbox(label="Overlapping speech", value=True)
    timestamps_cb = ft.Checkbox(label="Timestamps", value=False)
    disfluencies_cb = ft.Checkbox(label="Disfluencies", value=True)

    start_time = ft.TextField(
        label="Start", value="00:00:00", width=120, dense=True,
        input_filter=ft.InputFilter(r"[0-9:]"),
    )
    stop_time = ft.TextField(
        label="Stop", value="00:00:00", width=120, dense=True,
        input_filter=ft.InputFilter(r"[0-9:]"),
    )

    # ---- Log area -----------------------------------------------------
    log_list = ft.ListView(expand=True, spacing=2, auto_scroll=True)
    progress_bar = ft.ProgressBar(value=0, visible=False)

    def append_log(text: str, color: str = None):
        log_list.controls.append(
            ft.Text(text, size=13, color=color, selectable=True)
        )
        if len(log_list.controls) > 500:
            log_list.controls = log_list.controls[-400:]
        log_list.update()

    # Subscribe to events
    def on_log_event(payload):
        color = None
        level = payload.get("level", "")
        if level == "error":
            color = "#FF453A"
        elif level == "highlight":
            color = BRAND_BLUE
        append_log(payload.get("text", ""), color)

    def on_progress_event(payload):
        val = payload.get("value", 0)
        progress_bar.visible = True
        progress_bar.value = val / 100.0 if val > 1 else val
        progress_bar.update()

    state.bus.subscribe(EventType.LOG, on_log_event)
    state.bus.subscribe(EventType.PROGRESS, on_progress_event)

    # ---- Action buttons -----------------------------------------------
    def _build_job() -> TranscriptionJob:
        """Create a TranscriptionJob from current UI state."""
        audio = state.audio_files[0]
        transcript = state.transcript_files[0] if state.transcript_files else (
            os.path.splitext(audio)[0] + "." + (filetype_dropdown.value or "txt")
        )
        # Ensure model is downloaded
        sel_model = model_dropdown.value or "precise"
        if sel_model in ('fast', 'precise') and not model_manager.model_is_ready(sel_model):
            proxy_url, ignore_ssl = model_manager.get_proxy_from_config(
                {'proxy_url': get_config('proxy_url', ''),
                 'ignore_ssl': get_config('ignore_ssl', 'false')})
            append_log(f"Downloading '{sel_model}' model...", BRAND_BLUE)
            model_manager.download_model(sel_model, proxy_url=proxy_url, ignore_ssl=ignore_ssl)
        model_path = model_manager.get_model_path_for_app(sel_model) or sel_model

        import utils
        start_ms = utils.str_to_ms(start_time.value) if start_time.value and start_time.value != "00:00:00" else None
        stop_ms = utils.str_to_ms(stop_time.value) if stop_time.value and stop_time.value != "00:00:00" else None

        spk = speaker_dropdown.value or "auto"
        if spk == "off":
            spk = "none"

        return create_transcription_job(
            audio_file=audio,
            transcript_file=transcript,
            start_time=start_ms,
            stop_time=stop_ms,
            language_name=language_dropdown.value,
            whisper_model_name=model_path,
            speaker_detection=spk,
            overlapping=overlapping_cb.value,
            timestamps=timestamps_cb.value,
            disfluencies=disfluencies_cb.value,
            pause=pause_dropdown.value,
            languages=state.languages,
            get_config=get_config,
        )

    def _run_job_thread(job: TranscriptionJob):
        """Run transcription in a background thread."""
        from transcription_runner import run_transcription

        def log_fn(text, level='info'):
            color = None
            if level == 'error':
                color = "#FF453A"
            elif level == 'highlight':
                color = BRAND_BLUE
            try:
                log_list.controls.append(
                    ft.Text(text, size=13, color=color, selectable=True)
                )
                if len(log_list.controls) > 500:
                    log_list.controls = log_list.controls[-400:]
                log_list.update()
            except Exception:
                pass

        def progress_fn(pct):
            try:
                progress_bar.visible = True
                progress_bar.value = pct / 100.0
                progress_bar.update()
            except Exception:
                pass

        try:
            run_transcription(
                job=job,
                app_dir=state.app_dir,
                log_fn=log_fn,
                progress_fn=progress_fn,
                cancel_check=lambda: state.cancel,
            )
        finally:
            try:
                progress_bar.visible = False
                progress_bar.update()
                stop_btn.visible = False
                start_btn.disabled = False
                start_btn.update()
                stop_btn.update()
            except Exception:
                pass

    def on_start_click(e):
        if not state.audio_files:
            page.open(ft.SnackBar(ft.Text("Please select an audio file first.")))
            return
        try:
            job = _build_job()
        except Exception as ex:
            append_log(f"Error: {ex}", "#FF453A")
            return
        state.cancel = False
        start_btn.disabled = True
        stop_btn.visible = True
        start_btn.update()
        stop_btn.update()
        progress_bar.visible = True
        progress_bar.value = None
        progress_bar.update()
        append_log(f"Starting: {os.path.basename(job.audio_file)}", BRAND_BLUE)
        t = threading.Thread(target=_run_job_thread, args=(job,), daemon=True)
        t.start()

    def on_stop_click(e):
        state.cancel = True
        append_log("Canceling...", "#FF453A")

    def on_queue_click(e):
        if not state.audio_files:
            page.open(ft.SnackBar(ft.Text("Please select an audio file first.")))
            return
        append_log("Job added to queue.", BRAND_BLUE)

    start_btn = ft.ElevatedButton(
        "Start Transcription",
        icon=ft.Icons.PLAY_ARROW,
        color=ft.Colors.WHITE,
        bgcolor=BRAND_BLUE,
        on_click=on_start_click,
        width=220,
        height=44,
    )

    queue_btn = ft.OutlinedButton(
        "Add to Queue",
        icon=ft.Icons.ADD,
        on_click=on_queue_click,
        width=220,
    )

    stop_btn = ft.IconButton(
        icon=ft.Icons.STOP_CIRCLE_OUTLINED,
        tooltip="Stop transcription",
        icon_color="#FF453A",
        visible=False,
        on_click=on_stop_click,
    )

    # ---- Layout -------------------------------------------------------
    options_col = ft.Column(
        [
            ft.Text("New Transcription", size=20, weight=ft.FontWeight.BOLD),
            ft.Divider(height=8, color=ft.Colors.TRANSPARENT),

            # Audio file
            ft.Text("Audio file", size=12, weight=ft.FontWeight.W_500),
            ft.Row([
                ft.IconButton(
                    icon=ft.Icons.FOLDER_OPEN,
                    tooltip="Select audio file",
                    on_click=pick_audio,
                ),
                audio_file_text,
            ], spacing=4),

            # Transcript file
            ft.Text("Output file", size=12, weight=ft.FontWeight.W_500),
            ft.Row([
                ft.IconButton(
                    icon=ft.Icons.SAVE_AS,
                    tooltip="Set output file",
                    on_click=pick_transcript,
                ),
                transcript_file_text,
            ], spacing=4),

            ft.Divider(height=8, color=ft.Colors.TRANSPARENT),
            filetype_dropdown,
            ft.Row([start_time, stop_time], spacing=8),
            language_dropdown,
            model_dropdown,
            speaker_dropdown,
            pause_dropdown,
            ft.Divider(height=4, color=ft.Colors.TRANSPARENT),
            overlapping_cb,
            disfluencies_cb,
            timestamps_cb,
            ft.Divider(height=12, color=ft.Colors.TRANSPARENT),
            start_btn,
            queue_btn,
        ],
        spacing=6,
        width=300,
        scroll=ft.ScrollMode.AUTO,
    )

    # Right column: log output
    log_col = ft.Column(
        [
            ft.Row(
                [
                    ft.Text("Log", size=16, weight=ft.FontWeight.W_500),
                    ft.Row([stop_btn], spacing=4),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            progress_bar,
            ft.Container(
                content=log_list,
                expand=True,
                border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                border_radius=8,
                padding=10,
            ),
        ],
        expand=True,
        spacing=8,
    )

    return ft.Container(
        content=ft.Row(
            [
                options_col,
                ft.VerticalDivider(width=1),
                log_col,
            ],
            expand=True,
            spacing=16,
        ),
        padding=20,
        expand=True,
    )
