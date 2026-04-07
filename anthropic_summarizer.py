import os
import json
import traceback

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


def generate_meeting_summary(api_key: str, transcript_text: str,
                              model: str = None, system_prompt: str = None,
                              max_tokens: int = None, temperature: float = None) -> str:
    """
    Calls the Anthropic Claude API to generate a structured meeting summary.
    Falls back to config values, then defaults.
    """
    if not api_key:
        return "Error: No Anthropic API Key provided. Please add it in Settings."

    if not transcript_text or len(transcript_text.strip()) < 10:
        return "Error: The transcription is empty or too short to summarize."

    # Load from config if not provided
    try:
        from config import get_config
        if not model:
            model = get_config('summary_model', 'claude-sonnet-4-20250514')
        if not system_prompt:
            system_prompt = get_config('summary_prompt', DEFAULT_SUMMARY_PROMPT)
        if max_tokens is None:
            max_tokens = int(get_config('summary_max_tokens', 1500))
        if temperature is None:
            temperature = float(get_config('summary_temperature', 0.3))
    except Exception:
        model = model or 'claude-sonnet-4-20250514'
        system_prompt = system_prompt or DEFAULT_SUMMARY_PROMPT
        max_tokens = max_tokens or 1500
        temperature = temperature if temperature is not None else 0.3

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Here is the transcript:\n\n{transcript_text}"}
            ]
        )

        return response.content[0].text

    except ImportError:
        return (
            "Error: Anthropic library not installed.\n"
            "Please run 'pip install anthropic' in your terminal and restart the app."
        )
    except Exception as e:
        return f"Error connecting to Anthropic API:\n{str(e)}\n\n{traceback.format_exc()}"
