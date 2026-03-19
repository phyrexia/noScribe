import os
import json
import traceback

def generate_meeting_summary(api_key: str, transcript_text: str) -> str:
    """
    Calls the Anthropic Claude API to generate a structured meeting summary.
    """
    if not api_key:
        return "Error: No Anthropic API Key provided. Please add it in the Settings panel."
    
    if not transcript_text or len(transcript_text.strip()) < 10:
        return "Error: The transcription is empty or too short to summarize."

    system_prompt = (
        "You are an expert executive assistant and meeting scribe. "
        "Your task is to analyze the following meeting transcript and provide a highly structured, "
        "clear, and professional summary. Include:\n"
        "1. Executive Summary (2-3 sentences max)\n"
        "2. Key Discussion Points (Bullet points)\n"
        "3. Decisions Made (If any)\n"
        "4. Action Items (Assignee and Task, if identifiable)\n\n"
        "Only output the summary, nothing else. Provide the output in the same language as the transcript."
    )

    try:
        from anthropic import Anthropic
        
        client = Anthropic(api_key=api_key)
        
        # We use Claude 3.5 Sonnet for fast, cheap, and excellent reasoning
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            temperature=0.3,
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
