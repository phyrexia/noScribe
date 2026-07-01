# Visual Context for Meeting Summaries — Design

**Date:** 2026-07-01
**Branch:** `feat/visual-context` (placeholder; product rename pending — "ya no es MeetingGenie")
**Status:** Design, pending user review

## Problem

MeetingGenie accepts video files (`mp4`, `mov`, `mkv`, …) but `audio/convert.py`
extracts only the audio stream via PyAV — **the video frames are discarded**.
Summaries therefore miss everything shown on screen: slides, dashboards, code,
shared docs. A meeting where "the numbers on slide 4" drove a decision produces a
summary that never saw slide 4.

A separate project, `quantum-meetings`, already solves the visual half
(ffmpeg scene-detection keyframes → dedup → LLM vision → structured JSON →
timeline HTML). This feature ports that *logic* — not its Jitsi/GCP/rclone/Vigía
infrastructure — into MeetingGenie as an optional step.

## Goal

When the input is a video and the user opts in, extract representative keyframes,
describe each with a vision model, and fuse a timestamped visual-context block
into the existing Claude summary — so one summary reflects both what was said and
what was shown.

## Non-goals (YAGNI — deferred)

- Interactive keyframe/transcript timeline HTML (the `quantum-meetings` artifact).
- Per-frame JSON sidecars persisted to disk.
- Live recording / Jitsi capture.
- Making the **summary** provider-selectable — summary stays Claude (user decision:
  "split" — vision selectable, summary always Claude).
- Vigía-specific auth. (Vigía is reachable as a generic OpenAI-compatible endpoint;
  no special-casing.)

## Decisions locked with the user

1. **Outcome:** fuse visuals into the Claude summary (not a standalone timeline).
2. **Vision provider:** user-selectable, independent of the summary provider.
3. **Provider scope:** *split* — vision = Claude | OpenAI-compatible; summary = Claude.
4. **Custom/self-hosted models:** supported via a single **OpenAI-compatible** path
   (`{base_url, model, api_key}`) covering cloud OpenAI, LM Studio, Ollama, RunPod
   vLLM, and Vigía. Presets fill `base_url` + a default model.
5. **Keyframe engine:** PyAV (already bundled) — no ffmpeg CLI binary re-added.
6. **Default vision models:** `claude-haiku-4-5` / `gpt-4o-mini` (cheap tier).

## Architecture

### Flow

```
video → [existing] audio → transcript → diarize ─┐
                                                  ├─→ Claude summary (fused)
video has stream + toggle on →                    │
  keyframes → vision describe (per frame) →        │
  timestamped visual-context block ────────────────┘
audio-only input → visual step skipped entirely
```

Vision never blocks the summary: on any failure the summary proceeds
transcript-only with a one-line note.

### New modules (isolated, single-purpose)

**`keyframes.py`** — extraction, no LLM.
```python
@dataclass
class KeyframeCaps:
    sample_fps: float = 1.0       # decode cadence
    scene_threshold: float = 0.30 # mean abs grayscale diff vs last kept frame (0..1)
    min_gap_s: float = 4.0
    dhash_dist: int = 10          # perceptual-hash Hamming dedup distance
    max_frames: int = 40          # hard cap

@dataclass
class Keyframe:
    ts_ms: int
    path: str                     # written JPEG in a temp dir

def has_video_stream(media_path: str) -> bool
def extract_keyframes(media_path: str, caps: KeyframeCaps, tmp_dir: str) -> list[Keyframe]
```
PyAV opens the container, decodes the video stream at ~`sample_fps`, converts each
sampled frame to a small grayscale array (numpy), scores scene change as mean
absolute difference vs the last *kept* frame; a frame is kept when score >
`scene_threshold` **and** gap ≥ `min_gap_s`. Kept frames are deduped by dhash
(Pillow + numpy, computed inline — no `imagehash` dep) within `dhash_dist`; if the
count exceeds `max_frames`, keep the highest-scoring, roughly time-distributed
subset. JPEGs written via Pillow.

**`vision_providers.py`** — description, no extraction.
```python
class VisionClient(Protocol):
    def describe(self, image_bytes: bytes, prompt: str) -> dict: ...
    # returns {"tipo","resumen","texto_visible","hay_contenido_relevante"}

class ClaudeVision:        # anthropic SDK; model = claude-haiku-4-5
class OpenAICompatVision:  # openai SDK; base_url + model + api_key

def make_vision_client(config: dict) -> VisionClient | None
```
`OpenAICompatVision` covers cloud OpenAI **and** every local/self-hosted server
(LM Studio, Ollama, RunPod, Vigía) — they are all OpenAI chat-completions
compatible with base64 image content. Same structured-JSON contract as
`quantum-meetings` so prompts/outputs stay consistent.

**`visual_context.py`** — orchestration.
```python
def build_visual_context(media_path, client, caps, prompt, max_workers=4) -> str | None
```
extract → bounded `ThreadPoolExecutor` describe → drop frames with
`hay_contenido_relevante == false` → sort by `ts_ms` → render markdown:
`[mm:ss] <tipo> — <resumen>` (+ `texto visible: …` when present). Returns `None`
when there is no video stream or no relevant frames. Temp JPEGs deleted after.

### Changed files

**`anthropic_summarizer.py`** — add `visual_context: str | None = None`. When set,
prepend an `## On-screen content (by timestamp)` section before
`Here is the transcript:`. Backward compatible (defaults to `None`).

**`views/transcribe_page.py`** — in the Summarize handler: if
`has_video_stream(input)` and `visual_context_enabled`, run
`build_visual_context(...)` with progress feedback, pass result to the summarizer.

**`views/settings_page.py` + `config.py`** — new settings (below).

## Config keys (all in `~/.config/MeetingGenie/config.yml`)

| key | default | meaning |
|-----|---------|---------|
| `visual_context_enabled` | `true` | master toggle (auto-hidden for audio-only) |
| `vision_provider` | `"claude"` | `"claude"` \| `"openai_compatible"` |
| `vision_model_claude` | `"claude-haiku-4-5"` | Claude vision model |
| `vision_base_url` | `"https://api.openai.com/v1"` | OpenAI-compatible endpoint |
| `vision_model_openai` | `"gpt-4o-mini"` | model name on that endpoint |
| `vision_api_key` | `""` | key for the endpoint (blank OK for local) |
| `keyframe_scene_threshold` | `0.30` | advanced |
| `keyframe_min_gap_s` | `4.0` | advanced |
| `keyframe_max` | `40` | advanced |
| `keyframe_dhash_dist` | `10` | advanced |

Claude vision reuses the existing `anthropic_api_key`.

### Settings UX

- Toggle **"Include on-screen visuals in summary"**.
- **Vision provider** dropdown: Claude | OpenAI-compatible.
- When OpenAI-compatible: **preset** dropdown (OpenAI cloud / LM Studio / Ollama /
  RunPod-custom) that fills `vision_base_url` + a default model, plus editable
  base_url, model, and api_key fields.
- Advanced (collapsed): the four keyframe knobs.

## Cost / safety

Haiku / gpt-4o-mini vision + dedup + 40-frame hard cap → typically a few cents per
meeting on cloud; free on local endpoints. Bounded concurrency (4). Frames live in
a temp dir and are deleted after summarizing.

## Dependencies added to the bundle

- `openai` SDK (small, pure-python).
- PyAV, Pillow, numpy, anthropic — already bundled. **No ffmpeg binary.**
Add `openai` to `environments/requirements_macOS_arm64.txt` (and the other
`requirements_*` for parity) and to the PyInstaller hidden-imports if needed.

## Error handling

- No video stream → skip silently, summary as today.
- Missing/blank key for the chosen provider → skip visuals, summary transcript-only
  with a note in the UI.
- Per-frame vision error → skip that frame, continue.
- Local endpoint unreachable → skip visuals, clear UI error, summary proceeds.
- Vision **never** aborts the summary.

## Testing

- **Unit:** dhash dedup + cap selection on synthetic frame sequences;
  `make_vision_client` factory routing (claude vs base_url); `OpenAICompatVision`
  request shape against a stubbed SDK (asserts base64 image + model, no network);
  summarizer fusion with `visual_context` present/absent.
- **Integration (flagged, manual):** one short real clip end-to-end with a local
  LM Studio endpoint (no cloud cost) → assert a non-empty visual block reaches the
  summarizer.

## Future (out of scope, noted)

- Optional interactive timeline artifact (reuse `quantum-meetings` HTML).
- Persist keyframes + descriptions for re-summarization without re-running vision.
- Product rename + branch rename once the new name is chosen.
