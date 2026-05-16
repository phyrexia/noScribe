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
import wave
import contextlib


# Punctuation that ends a sentence. Includes Spanish/English/East-Asian variants.
_SENTENCE_TERMINATORS = (".", "?", "!", "…", "。", "？", "！")

# Soft break punctuation (clause boundaries). When the buffer is long enough,
# we treat these as acceptable flush points so the transcript reads as
# chunked clauses instead of unbroken paragraphs — Apple Speech often omits
# full stops on conversational audio but does insert commas/colons.
_SOFT_BREAK_PUNCT = (",", ";", ":", "，", "；", "：")

# Minimum words needed before a soft-break is allowed to flush. Prevents
# tiny "yes,"/"no," fragments.
_SOFT_BREAK_MIN_WORDS = 8

# Hard cap on words per sentence — keeps un-punctuated streams readable.
_MAX_WORDS_PER_SENTENCE = 30

# Inter-word gap (seconds) that forces a sentence flush. Rarely fires because
# SFSpeechRecognizer returns 0.0 timestamps on file-URL requests (see TODO
# below), but kept as a safety net in case Apple ever fixes the bug.
_GAP_FLUSH_SECONDS = 0.8


def _read_wav_duration_seconds(path: str) -> float:
    """Return the duration of a WAV file in seconds, or 0.0 on failure.

    The transcription runner always decodes to 16 kHz mono WAV before invoking
    a worker, so `wave` from the stdlib is sufficient — no extra deps needed.
    """
    try:
        with contextlib.closing(wave.open(path, "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            return frames / float(rate)
    except Exception:
        return 0.0


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


def _words_with_timestamps(raw_words, total_duration_s: float):
    """Materialize {text, start, end} dicts from accumulated partial-result words.

    `raw_words` is a list of `{"text": str, "_ts": float, "_dur": float}` dicts
    captured while pumping partial results.

    Apple Speech's `SFTranscriptionSegment.timestamp()` and `.duration()`
    return 0.0 for every word when the request is built from a file URL — a
    framework quirk documented at https://developer.apple.com/forums/thread/118325
    (also reported as FB8961064). When at least half of the words have real
    timestamps we use them; otherwise we synthesize by distributing words
    uniformly across the known audio duration. The synthetic approximation is
    coarse (it ignores silence and per-word variation) but sufficient for
    pyannote's `find_speaker` to map each sentence onto a diarization slot,
    since pyannote segments are typically multi-second.

    TODO(apple-speech-timestamps): switch to real timestamps as soon as Apple
    fixes SFSpeechRecognitionResult to populate `.timestamp()` for file URLs.
    """
    if not raw_words:
        return []

    real_ts_count = sum(1 for w in raw_words if w["_ts"] > 0.0)
    use_real = real_ts_count >= max(1, int(0.5 * len(raw_words)))

    if use_real:
        return [
            {
                "text": w["text"],
                "start": w["_ts"],
                "end": w["_ts"] + max(w["_dur"], 0.0),
            }
            for w in raw_words
        ]

    if total_duration_s <= 0:
        # Last resort: 0.5s/word so segments at least carry distinct,
        # non-zero times — pyannote labeling will be approximate but consistent.
        return [
            {"text": w["text"], "start": i * 0.5, "end": (i + 1) * 0.5}
            for i, w in enumerate(raw_words)
        ]

    N = len(raw_words)
    per = total_duration_s / float(N)
    return [
        {"text": w["text"], "start": i * per, "end": (i + 1) * per}
        for i, w in enumerate(raw_words)
    ]


def _join_words(buf):
    """Pretty-print a list of word dicts as a sentence.

    Apple Speech emits punctuation as standalone tokens (e.g. `["lado", ","]`),
    so a naive space-join produces `"lado ,"`. We attach trailing punctuation
    to the preceding word and strip the leading space for opening Spanish
    punctuation (¿ ¡).
    """
    parts = []
    for w in buf:
        tok = w["text"]
        if not tok:
            continue
        # Closing punctuation attaches to previous token.
        if parts and tok and tok[0] in ",.;:?!…)»”’%":
            parts[-1] = parts[-1] + tok
        elif parts and parts[-1] and parts[-1][-1] in "¿¡(«“‘":
            # Opening punctuation: don't insert a space between it and word.
            parts[-1] = parts[-1] + tok
        else:
            parts.append(tok)
    return " ".join(parts).strip()


def _aggregate_sentences(words):
    """Group word-level entries into sentence-level segments.

    Flush conditions, in order of priority:
      1. The current word ends with sentence-terminating punctuation
         (`. ? ! …`). This is the strongest signal.
      2. The current word is/ends with soft-break punctuation (`, ; :`) AND
         the buffer already has at least _SOFT_BREAK_MIN_WORDS words. Apple
         Speech on conversational audio frequently omits full stops but does
         insert commas, so commas double as clause boundaries — without them
         the transcript reads as huge unbroken paragraphs.
      3. The inter-word gap to the next word exceeds _GAP_FLUSH_SECONDS.
         Rarely fires because Apple Speech returns 0.0 timestamps on file
         URLs (see `_words_with_timestamps`), but kept as a safety net.
      4. The buffer reaches _MAX_WORDS_PER_SENTENCE (prevents unbounded
         paragraphs when the recognizer omits punctuation entirely).
    """
    if not words:
        return []

    sentences = []
    buf = []  # list of word dicts

    def _flush():
        if not buf:
            return
        text = _join_words(buf)
        if not text:
            buf.clear()
            return
        sentences.append({
            "start": buf[0]["start"],
            "end": buf[-1]["end"],
            "text": text,
        })
        buf.clear()

    def _flush_at_last_punct():
        """If the buffer contains a soft-break token, flush up to and including
        that token and keep the remainder in `buf`. Falls back to flushing the
        whole buffer if no punctuation is found.
        """
        # Scan from the end so we cut as late as possible (closest to the cap).
        for j in range(len(buf) - 1, max(_SOFT_BREAK_MIN_WORDS - 1, 0) - 1, -1):
            tok = buf[j]["text"].rstrip()
            if tok and tok[-1] in (_SENTENCE_TERMINATORS + _SOFT_BREAK_PUNCT):
                head = buf[:j + 1]
                tail = buf[j + 1:]
                text = _join_words(head)
                if text:
                    sentences.append({
                        "start": head[0]["start"],
                        "end": head[-1]["end"],
                        "text": text,
                    })
                buf.clear()
                buf.extend(tail)
                return
        _flush()

    for i, w in enumerate(words):
        buf.append(w)
        token = w["text"].rstrip()
        last_char = token[-1] if token else ""
        ends_sentence = last_char in _SENTENCE_TERMINATORS
        is_soft_break = (
            last_char in _SOFT_BREAK_PUNCT
            and len(buf) >= _SOFT_BREAK_MIN_WORDS
        )

        next_gap = 0.0
        if i + 1 < len(words):
            next_gap = max(0.0, float(words[i + 1]["start"]) - float(w["end"]))

        if ends_sentence or is_soft_break or next_gap > _GAP_FLUSH_SECONDS:
            _flush()
        elif len(buf) >= _MAX_WORDS_PER_SENTENCE:
            # Hit the hard cap. Try to break at the last punctuation in the
            # buffer so the cut lands on a clause boundary instead of mid-phrase.
            _flush_at_last_punct()

    _flush()
    return sentences


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
        # Partial results MUST be enabled: SFSpeechRecognizer processes audio
        # in ~1-minute windows internally, and the final result only contains
        # the LAST window's transcription. The full transcript is reconstructed
        # by accumulating new words from each partial as it arrives.
        request.setShouldReportPartialResults_(True)
        try:
            # Dictation hint biases the model toward connected speech. Falls
            # back silently if the constant is missing in an older SDK.
            request.setTaskHint_(Speech.SFSpeechRecognitionTaskHintDictation)
        except Exception:
            pass
        try:
            # Automatic punctuation (macOS 13+/iOS 16+). Without this the
            # recognizer returns a stream of un-capitalized words with no
            # sentence boundaries, which makes the transcript unreadable and
            # defeats the sentence aggregator below.
            request.setAddsPunctuation_(True)
        except Exception:
            pass

        # Audio duration — used to synthesize timestamps when Apple returns 0.0
        # for every word (see TODO below in _extract_words).
        total_duration_s = _read_wav_duration_seconds(audio_path)
        if total_duration_s <= 0:
            plog("info", "Could not determine audio duration; synthetic timestamps will be 0.")
        else:
            plog("debug", f"Audio duration: {total_duration_s:.2f}s")

        plog("info", f"Starting on-device Apple Speech recognition ({locale_id}).")

        done = threading.Event()
        state = {
            "error": None,               # NSError → str
            "accumulated_words": [],     # raw word dicts collected across partials
            "consumed": 0,               # index into the current partial's segments
        }

        def _consume_partial(transcription):
            """Append new segments from `transcription` to accumulated_words.

            Each partial result re-emits the entire transcription-so-far, so we
            slice from `state['consumed']` to capture only the new tail.
            """
            try:
                segs = transcription.segments()
            except Exception:
                return
            if segs is None:
                return
            count = segs.count() if hasattr(segs, "count") else len(segs)
            for i in range(state["consumed"], count):
                seg = segs[i]
                try:
                    text = str(seg.substring())
                    ts = float(seg.timestamp())
                    dur = float(seg.duration())
                except Exception:
                    continue
                state["accumulated_words"].append({
                    "text": text, "_ts": ts, "_dur": dur,
                })
            state["consumed"] = count

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
                if is_final:
                    # The final result resets to the last window only; ignore
                    # its segments and rely on what we accumulated from partials.
                    done.set()
                else:
                    _consume_partial(transcription)
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
            # Rough progress heuristic based on elapsed wall-clock time —
            # Apple Speech doesn't expose a per-file progress callback, so we
            # clamp at 95% and let the post-processing step bump it to 100.
            pct = min(95, int(elapsed))
            if pct != last_progress:
                try:
                    q.put({"type": "progress", "pct": pct})
                except Exception:
                    pass
                last_progress = pct

        if state["error"]:
            raise RuntimeError(f"Apple Speech error: {state['error']}")

        raw_words = state["accumulated_words"]
        if not raw_words:
            raise RuntimeError("Apple Speech finished with no transcription words.")

        # Synthesize timestamps if the framework returned 0.0 for every word
        # (see _extract_words for the file-URL timestamp bug), then aggregate
        # into sentence-level segments by punctuation.
        words = _words_with_timestamps(raw_words, total_duration_s)
        sentences = _aggregate_sentences(words)

        plog("debug", f"Apple Speech emitted {len(sentences)} sentence(s) from {len(words)} word(s).")

        for sent in sentences:
            try:
                q.put({
                    "type": "segment",
                    "segment": {
                        "start": sent["start"],
                        "end": sent["end"],
                        "text": sent["text"],
                    },
                })
            except Exception:
                pass

        approx_duration = sentences[-1]["end"] if sentences else total_duration_s

        try:
            q.put({"type": "progress", "pct": 100})
        except Exception:
            pass

        info_dict = {
            "language": locale_id,
            "language_probability": 1.0,
            "duration": approx_duration,
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
