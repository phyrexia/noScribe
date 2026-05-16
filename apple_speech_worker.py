# MeetingGenie - Apple Speech Recognition worker (on-device).
#
# Uses SFSpeechRecognizer via pyobjc-framework-Speech. Runs entirely on the
# Apple Neural Engine — no network, no quota — and is gated to on-device
# recognition (`requiresOnDeviceRecognition = True`) for privacy.
#
# Output mirrors the existing whisper workers' queue protocol so the runner
# can consume it interchangeably:
#   {"type": "device", ...}
#   {"type": "log", "level": ..., "msg": ...}
#   {"type": "progress", "pct": ...}
#   {"type": "segment", "segment": {"start": s, "end": s, "text": str}}
#   {"type": "result", "ok": True|False, "info"|"error"|"trace": ...}
#   {"type": "finished"}

import os
import threading
import traceback


# ISO 639-1 → BCP-47 locale used by SFSpeechRecognizer. Apple has a fixed
# set of supportedLocales(); we map MeetingGenie's whisper language codes
# to a reasonable default region.
LOCALE_MAP = {
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "pt": "pt-BR",
    "ja": "ja-JP",
    "zh": "zh-CN",
    "ko": "ko-KR",
    "ru": "ru-RU",
    "ar": "ar-SA",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "tr": "tr-TR",
    "sv": "sv-SE",
    "da": "da-DK",
    "fi": "fi-FI",
    "nb": "nb-NO",
    "no": "nb-NO",
    "cs": "cs-CZ",
    "el": "el-GR",
    "he": "he-IL",
    "hi": "hi-IN",
    "id": "id-ID",
    "ms": "ms-MY",
    "ro": "ro-RO",
    "sk": "sk-SK",
    "th": "th-TH",
    "uk": "uk-UA",
    "vi": "vi-VN",
    "ca": "ca-ES",
    "hr": "hr-HR",
    "hu": "hu-HU",
}


def _map_locale(language_code: str) -> str:
    if not language_code:
        return "en-US"
    code = language_code.strip()
    if "-" in code or "_" in code:
        # Already looks like a BCP-47 identifier.
        return code.replace("_", "-")
    return LOCALE_MAP.get(code.lower(), "en-US")


def _request_authorization(timeout_s: float = 60.0):
    """Block until SFSpeechRecognizer authorization status is known.

    Returns the integer status (SFSpeechRecognizerAuthorizationStatus*):
      0 NotDetermined, 1 Denied, 2 Restricted, 3 Authorized.
    Raises RuntimeError on timeout.
    """
    import Speech

    done = threading.Event()
    result = {"status": None}

    def _cb(status):
        result["status"] = int(status)
        done.set()

    Speech.SFSpeechRecognizer.requestAuthorization_(_cb)
    if not done.wait(timeout=timeout_s):
        raise RuntimeError("Speech recognition authorization timed out")
    return result["status"]


def apple_speech_proc_entrypoint(args: dict, q):
    """Entry point for the Apple Speech recognition subprocess.

    Expected args:
      audio_path     : str, path to a decoded WAV (PyAV step output).
      language_name  : str, friendly language name (e.g. "Spanish") or "Auto".
      language_code  : str | None, ISO 639-1 (e.g. "es") or BCP-47 ("es-ES").
      locale         : str, UI locale for i18n (not the recognition locale).
    """
    # The runner pre-sets these for the offline pyannote path; clear them so
    # any Apple Speech model assets (rare; usually preinstalled) can fetch.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    try:
        q.put({
            "type": "device",
            "role": "transcription",
            "backend": "apple-ane",
            "label": "Apple Neural Engine",
        })
    except Exception:
        pass

    def plog(level, msg):
        try:
            q.put({"type": "log", "level": level, "msg": str(msg)})
        except Exception:
            pass

    try:
        import Speech
        from Foundation import NSURL, NSLocale

        audio_path = args.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio path does not exist: {audio_path}")

        language_name = args.get("language_name")
        language_code = args.get("language_code")
        if language_name == "Multilingual" or language_name == "Auto" or not language_code:
            # Apple Speech requires a concrete locale. Default to en-US when
            # auto-detection is requested (Apple has no language ID API).
            locale_id = "en-US"
        else:
            locale_id = _map_locale(language_code)

        # Authorization. The first call shows a TCC prompt; subsequent calls
        # return cached status immediately.
        plog("info", f"Requesting Speech recognition authorization for locale {locale_id}...")
        auth_status = _request_authorization()
        if auth_status != 3:
            status_names = {0: "not determined", 1: "denied", 2: "restricted", 3: "authorized"}
            name = status_names.get(auth_status, str(auth_status))
            raise RuntimeError(
                f"Speech recognition not authorized (status: {name}). "
                f"Open System Settings → Privacy & Security → Speech Recognition "
                f"and enable MeetingGenie."
            )

        locale = NSLocale.localeWithLocaleIdentifier_(locale_id)
        recognizer = Speech.SFSpeechRecognizer.alloc().initWithLocale_(locale)
        if recognizer is None:
            raise RuntimeError(f"SFSpeechRecognizer init failed for locale {locale_id}")
        if not recognizer.isAvailable():
            raise RuntimeError(f"Speech recognition unavailable for locale {locale_id}")
        if not recognizer.supportsOnDeviceRecognition():
            raise RuntimeError(
                f"On-device recognition not supported for locale {locale_id}. "
                f"MeetingGenie refuses to fall back to server-based mode."
            )

        url = NSURL.fileURLWithPath_(audio_path)
        request = Speech.SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
        request.setRequiresOnDeviceRecognition_(True)
        request.setShouldReportPartialResults_(True)
        try:
            # Dictation hint biases the model toward connected speech. Falls
            # back silently if the constant is missing in an older SDK.
            request.setTaskHint_(Speech.SFSpeechRecognitionTaskHintDictation)
        except Exception:
            pass

        plog("info", f"Starting on-device Apple Speech recognition ({locale_id}).")

        done = threading.Event()
        state = {
            "emitted": 0,        # number of segments already sent to the queue
            "error": None,       # NSError → str
            "final_segments": [],
            "approx_duration": 0.0,
        }

        def _emit_new_segments(transcription, is_final: bool):
            """Send segments[state['emitted']:] to the queue."""
            segs = transcription.segments()
            if segs is None:
                return
            count = segs.count() if hasattr(segs, "count") else len(segs)
            for i in range(state["emitted"], count):
                seg = segs[i]
                try:
                    text = str(seg.substring())
                    start = float(seg.timestamp())
                    dur = float(seg.duration())
                except Exception:
                    continue
                end = start + dur
                state["approx_duration"] = max(state["approx_duration"], end)
                try:
                    q.put({
                        "type": "segment",
                        "segment": {"start": start, "end": end, "text": text},
                    })
                except Exception:
                    pass
                if is_final:
                    state["final_segments"].append({
                        "start": start, "end": end, "text": text,
                    })
            state["emitted"] = count

        def _result_handler(result, error):
            try:
                if error is not None:
                    state["error"] = str(error.localizedDescription())
                    done.set()
                    return
                if result is None:
                    return
                transcription = result.bestTranscription()
                if transcription is None:
                    return
                is_final = bool(result.isFinal())
                _emit_new_segments(transcription, is_final)
                if is_final:
                    done.set()
            except Exception as cb_err:
                state["error"] = f"callback error: {cb_err}"
                done.set()

        task = recognizer.recognitionTaskWithRequest_resultHandler_(
            request, _result_handler
        )
        if task is None:
            raise RuntimeError("Failed to start Apple Speech recognition task")

        # Poll the run-loop equivalent: the Speech framework delivers callbacks
        # on its own dispatch queue, so the worker can simply block on `done`
        # while emitting heartbeat progress.
        # Apple Speech delivers callbacks via a dispatch queue tied to the
        # main run loop. Without a running CFRunLoop the result handler never
        # fires, so we pump the run loop in slices instead of plain sleeping.
        from Foundation import NSRunLoop, NSDate
        run_loop = NSRunLoop.currentRunLoop()

        last_progress = -1
        max_wait_s = 60 * 60
        elapsed = 0.0
        while not done.is_set():
            # Pump the run loop for 0.5s so the Speech framework can deliver
            # partial and final result callbacks on this thread.
            run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.5))
            elapsed += 0.5
            if done.is_set():
                break
            if elapsed >= max_wait_s:
                try:
                    task.cancel()
                except Exception:
                    pass
                raise RuntimeError("Apple Speech recognition timed out (>1h).")
            # Rough progress heuristic: ratio of last emitted segment end to
            # the file's duration, if we have one. Otherwise nudge based on
            # elapsed seconds.
            pct = 0
            if state["approx_duration"] > 0:
                # We don't always know the total duration here; clamp to 95.
                pct = min(95, int(state["approx_duration"] * 0.9))
            else:
                pct = min(90, int(elapsed))
            if pct != last_progress:
                try:
                    q.put({"type": "progress", "pct": pct})
                except Exception:
                    pass
                last_progress = pct

        if state["error"]:
            raise RuntimeError(f"Apple Speech error: {state['error']}")

        try:
            q.put({"type": "progress", "pct": 100})
        except Exception:
            pass

        info_dict = {
            "language": locale_id,
            "language_probability": 1.0,
            "duration": state["approx_duration"],
        }
        try:
            q.put({"type": "result", "ok": True, "info": info_dict})
        except Exception:
            pass
        try:
            q.put({"type": "finished"})
        except Exception:
            pass
        plog("debug", "Apple Speech subprocess finished cleanly.")

    except Exception as e:
        try:
            q.put({
                "type": "result",
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(),
            })
        except Exception:
            pass
        try:
            q.put({"type": "finished"})
        except Exception:
            pass
