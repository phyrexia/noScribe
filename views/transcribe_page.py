# MeetingGenie - Transcribe Page
# File selection, options, start button, log output, progress bar

import os
import platform
import threading
import multiprocessing as mp
import queue as pyqueue
import flet as ft
from app_state import AppState
from event_bus import EventType
from models import TranscriptionJob, create_transcription_job
from config import get_config
import model_manager

BRAND_BLUE = "#0A84FF"


def _live_target(args, q, stop_event):
    """Wrapper to import live worker only in subprocess. Must be module-level for pickle."""
    from live_mp_worker import live_proc_entrypoint
    live_proc_entrypoint(args, q, stop_event)


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

    def on_filetype_change(e):
        """Update transcript filename extension when format changes."""
        if state.transcript_files:
            base = os.path.splitext(state.transcript_files[0])[0]
            ext = filetype_dropdown.value or "txt"
            state.transcript_files = [f"{base}.{ext}"]
            transcript_file_text.value = os.path.basename(state.transcript_files[0])
            transcript_file_text.italic = False
            page.update()

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
        on_select=on_filetype_change,
    )

    # Audio input devices for Live Mode
    def _get_input_devices():
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            return [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        except Exception:
            return []

    input_devices = _get_input_devices()
    input_device_dropdown = ft.Dropdown(
        label="Audio input (Live Mode)",
        width=280,
        dense=True,
        options=[ft.dropdown.Option(str(i), name) for i, name in input_devices],
        value=str(input_devices[0][0]) if input_devices else None,
    )

    overlapping_cb = ft.Checkbox(label="Overlapping speech", value=True)
    timestamps_cb = ft.Checkbox(label="Timestamps", value=False)
    disfluencies_cb = ft.Checkbox(label="Disfluencies", value=True)

    def _on_api_key_blur(e):
        """Auto-save API key when field loses focus."""
        val = anthropic_key.value.strip()
        if val:
            state.set_config('anthropic_api_key', val)
            state.save_config()

    anthropic_key = ft.TextField(
        label="Anthropic API Key",
        value=state.get_config('anthropic_api_key', ''),
        password=True,
        can_reveal_password=True,
        width=220,
        dense=True,
        on_blur=_on_api_key_blur,
    )

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
        ext = filetype_dropdown.value or "txt"
        # Always use the currently selected format, not what was auto-set
        if state.transcript_files:
            base = os.path.splitext(state.transcript_files[0])[0]
            transcript = f"{base}.{ext}"
        else:
            transcript = os.path.splitext(audio)[0] + f".{ext}"
        # Update display
        transcript_file_text.value = os.path.basename(transcript)
        transcript_file_text.italic = False
        state.transcript_files = [transcript]
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

    # ---- Speaker naming bridge (worker↔UI) ----
    from views.dialogs.speaker_naming import SpeakerNamingBridge
    _speaker_bridge = SpeakerNamingBridge()
    _speaker_bridge.setup(page)
    page.pubsub.subscribe(_speaker_bridge.on_pubsub_message)

    def _run_job_thread(job: TranscriptionJob):
        """Run transcription in a background thread."""
        from transcription_runner import run_transcription

        def speaker_naming_fn(speakers_data, audio_path):
            """Called from worker thread — uses pubsub bridge to show dialog on UI thread."""
            return _speaker_bridge.request_naming(speakers_data, audio_path, app_dir=state.app_dir)

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
                speaker_naming_fn=speaker_naming_fn,
            )
            # Notify editor to open transcript
            if job.transcript_file and os.path.exists(job.transcript_file):
                state.bus.publish(EventType.JOB_FINISHED, {
                    "transcript_path": job.transcript_file
                })
        finally:
            try:
                progress_bar.visible = False
                progress_bar.update()
                stop_btn.visible = False
                start_btn.disabled = False
                summarize_btn.disabled = False
                start_btn.update()
                stop_btn.update()
                summarize_btn.update()
            except Exception:
                pass

    def on_start_click(e):
        if not state.audio_files:
            page.show_dialog(ft.SnackBar(ft.Text("Please select an audio file first.")))
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
            page.show_dialog(ft.SnackBar(ft.Text("Please select an audio file first.")))
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

    def on_summarize_click(e):
        key = (anthropic_key.value or "").strip()
        if not key:
            # Fallback to config
            key = (state.get_config('anthropic_api_key', '') or "").strip()
        if not key:
            append_log("Please enter your Anthropic API Key in Settings or in the field below.", "#FF453A")
            return
        # Save key to config
        state.set_config('anthropic_api_key', key)
        state.save_config()
        # Get transcript text from last job
        if not state.transcript_files:
            append_log("No transcript file to summarize.", "#FF453A")
            return
        transcript_path = state.transcript_files[0]
        if not os.path.exists(transcript_path):
            append_log(f"File not found: {transcript_path}", "#FF453A")
            return
        append_log("Generating AI summary...", BRAND_BLUE)
        summarize_btn.disabled = True
        summarize_btn.update()

        def _do_summarize():
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                from anthropic_summarizer import generate_meeting_summary
                result = generate_meeting_summary(key, text)

                # Save as .md next to transcript
                md_path = os.path.splitext(transcript_path)[0] + "_summary.md"
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Meeting Summary\n\n")
                    f.write(f"**Source:** {os.path.basename(transcript_path)}\n\n")
                    f.write(result)
                state.last_summary_path = md_path

                append_log("", None)
                append_log("━━━ AI Summary ━━━", BRAND_BLUE)
                for line in result.split('\n'):
                    append_log(line, None)
                append_log("━━━━━━━━━━━━━━━━━━", BRAND_BLUE)
                append_log(f"Saved: {md_path}", BRAND_BLUE)

                # Signal editor to open the summary
                state.bus.publish(EventType.JOB_FINISHED, {"summary_path": md_path, "transcript_path": transcript_path})
            except Exception as ex:
                append_log(f"Summary error: {ex}", "#FF453A")
            finally:
                try:
                    summarize_btn.disabled = False
                    summarize_btn.update()
                except Exception:
                    pass

        threading.Thread(target=_do_summarize, daemon=True).start()

    summarize_btn = ft.ElevatedButton(
        "Generar Resumen",
        icon=ft.Icons.AUTO_AWESOME,
        on_click=on_summarize_click,
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
            ft.Divider(height=8, color=ft.Colors.TRANSPARENT),
            anthropic_key,
            ft.Divider(height=8, color=ft.Colors.TRANSPARENT),
            input_device_dropdown,
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
                    ft.Row([summarize_btn, stop_btn], spacing=4),
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

    # ---- Live Mode -------------------------------------------------------
    live_output = ft.TextField(
        multiline=True,
        read_only=True,
        min_lines=20,
        expand=True,
        text_size=18,
        value="Listening for audio...\n\n",
        border=ft.InputBorder.NONE,
        content_padding=ft.padding.all(16),
    )

    live_col = ft.Column(
        [
            ft.Row(
                [
                    ft.Row([
                        ft.Icon(ft.Icons.FIBER_MANUAL_RECORD, color="#FF453A", size=14),
                        ft.Text("Live Transcription", size=16, weight=ft.FontWeight.W_500),
                    ], spacing=6),
                ],
            ),
            ft.Container(
                content=live_output,
                expand=True,
                border=ft.border.all(1, "#FF453A"),
                border_radius=8,
            ),
        ],
        expand=True,
        spacing=8,
        visible=False,
    )

    # Right panel switches between log and live
    right_panel = ft.Column(
        [log_col, live_col],
        expand=True,
    )

    def _poll_live_queue(live_q, stop_evt, proc):
        """Poll live queue in a thread and push segments to UI."""
        while not stop_evt.is_set():
            try:
                msg = live_q.get(timeout=0.5)
            except pyqueue.Empty:
                # Check if process died
                if not proc.is_alive():
                    exitcode = proc.exitcode
                    append_log(f"[Live] Worker exited unexpectedly (code {exitcode})", "#FF453A")
                    break
                continue
            except Exception:
                break

            mtype = msg.get("type")
            if mtype == "live_segment":
                text = msg.get("text", "")
                try:
                    live_output.value = (live_output.value or "") + f"- {text}\n"
                    live_output.update()
                except Exception:
                    pass
            elif mtype == "log":
                lvl = msg.get("level", "info")
                color = "#FF453A" if lvl == "error" else None
                try:
                    live_output.value = (live_output.value or "") + f"[{lvl}] {msg.get('msg', '')}\n"
                    live_output.update()
                except Exception:
                    pass
                append_log(f"[Live] {msg.get('msg', '')}", color)
            elif mtype == "live_finished":
                break

        state.live_process_running = False
        try:
            live_btn.text = "Live Mode"
            live_btn.bgcolor = "#D84315"
            live_btn.update()
            live_col.visible = False
            log_col.visible = True
            right_panel.update()
        except Exception:
            pass

    def _toggle_live(e):
        if state.live_process_running:
            # Stop
            if hasattr(state, '_live_stop_event'):
                state._live_stop_event.set()
            return

        # Start
        state.live_process_running = True
        live_btn.text = "Stop Live"
        live_btn.bgcolor = "#C62828"
        live_btn.update()

        # Show live panel, hide log
        device_name = "default"
        for i, name in input_devices:
            if str(i) == str(input_device_dropdown.value):
                device_name = name
                break
        live_output.value = f"Starting live transcription...\nDevice: {device_name}\nLoading models (this may take a moment)...\n\n"
        live_col.visible = True
        log_col.visible = False
        right_panel.update()

        # Prepare args
        sel_model = model_dropdown.value or "fast"
        model_path = model_manager.get_model_path_for_app(sel_model) or sel_model

        language_name = language_dropdown.value or "Auto"
        language_code = None
        if language_name not in ('Auto', 'Multilingual'):
            language_code = state.languages.get(language_name)

        proxy_url = get_config('proxy_url', '')
        ignore_ssl = get_config('ignore_ssl', 'false').lower() == 'true'

        # SSL env vars for child process
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if ignore_ssl:
            os.environ["CURL_CA_BUNDLE"] = ""
            os.environ["REQUESTS_CA_BUNDLE"] = ""

        args = {
            "locale": get_config('locale', 'en'),
            "device": "auto" if platform.system() == "Darwin" else "cpu",
            "compute_type": get_config('whisper_compute_type', 'int8'),
            "model_name_or_path": model_path,
            "input_device_id": input_device_dropdown.value,
            "language_code": language_code,
            "language_name": language_name,
            "vad_threshold": float(get_config('vad_threshold', 0.5)),
            "cpu_threads": int(get_config('threads', 4)),
            "ignore_ssl": ignore_ssl,
            "proxy_url": proxy_url,
        }

        ctx = mp.get_context("spawn")
        live_q = ctx.Queue()
        stop_evt = ctx.Event()
        state._live_stop_event = stop_evt

        proc = ctx.Process(target=_live_target, args=(args, live_q, stop_evt), daemon=True)
        proc.start()
        append_log(f"Live worker started (device: {input_device_dropdown.value})", BRAND_BLUE)

        threading.Thread(target=_poll_live_queue, args=(live_q, stop_evt, proc), daemon=True).start()

    live_btn = ft.ElevatedButton(
        "Live Mode",
        icon=ft.Icons.MIC,
        bgcolor="#D84315",
        color=ft.Colors.WHITE,
        on_click=_toggle_live,
    )
    # Expose to shell header
    state.live_btn = live_btn

    return ft.Container(
        content=ft.Row(
            [
                options_col,
                ft.VerticalDivider(width=1),
                right_panel,
            ],
            expand=True,
            spacing=16,
        ),
        padding=20,
        expand=True,
    )
