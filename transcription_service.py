# MeetingGenie - Transcription Service
# Helper functions extracted from noScribe.py and interface for Flet migration.
#
# Phase 0: standalone helpers (overlap_len, find_speaker, html helpers)
# Phase 2: full _process_single_job migration with event bus callbacks

import os
import re
import html
from pathlib import Path

import AdvancedHTMLParser

import utils

# --- Timestamp regex --------------------------------------------------

timestamp_re = re.compile(r'\[\d\d:\d\d:\d\d.\d\d\d --> \d\d:\d\d:\d\d.\d\d\d\]')

# --- Default HTML template --------------------------------------------

DEFAULT_HTML = """
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">
<html >
<head >
<meta charset="UTF-8" />
<meta name="qrichtext" content="1" />
<style type="text/css" >
p, li { white-space: pre-wrap; }
</style>
<style type="text/css" >
 p { font-size: 0.9em; }
 .MsoNormal { font-family: "Arial"; font-weight: 400; font-style: normal; font-size: 0.9em; }
 @page WordSection1 {mso-line-numbers-restart: continuous; mso-line-numbers-count-by: 1; mso-line-numbers-start: 1; }
 div.WordSection1 {page:WordSection1;}
</style>
</head>
<body style="font-family: 'Arial'; font-weight: 400; font-style: normal" >
</body>
</html>"""

# --- Iterator helper --------------------------------------------------

def iter_except(function, exception):
    """Works like builtin 2-argument `iter()`, but stops on `exception`."""
    try:
        while True:
            yield function()
    except exception:
        return

# --- HTML helpers -----------------------------------------------------

def html_node_to_text(node: AdvancedHTMLParser.AdvancedTag) -> str:
    """Recursively get all text from an HTML node and its children."""
    if AdvancedHTMLParser.isTextNode(node):
        return html.unescape(node)
    elif AdvancedHTMLParser.isTagNode(node):
        text_parts = []
        for child in node.childBlocks:
            text = html_node_to_text(child)
            if text:
                text_parts.append(text)
        if node.tagName.lower() in ['p', 'div', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'br']:
            if node.tagName.lower() == 'br':
                return '\n'
            else:
                return '\n' + ''.join(text_parts).strip() + '\n'
        else:
            return ''.join(text_parts)
    else:
        return ''


def html_to_text(parser: AdvancedHTMLParser.AdvancedHTMLParser) -> str:
    return html_node_to_text(parser.body)


def vtt_escape(txt: str) -> str:
    txt = html.escape(txt, quote=False)
    while txt.find('\n\n') > -1:
        txt = txt.replace('\n\n', '\n')
    return txt


def html_to_webvtt(parser: AdvancedHTMLParser.AdvancedHTMLParser, media_path: str) -> str:
    vtt = 'WEBVTT '
    paragraphs = parser.getElementsByTagName('p')
    vtt += vtt_escape(paragraphs[0].textContent) + '\n\n'
    vtt += vtt_escape('NOTE\n' + html_node_to_text(paragraphs[1])) + '\n\n'
    vtt += f'NOTE media: {media_path}\n\n'

    segments = parser.getElementsByTagName('a')
    for i in range(len(segments)):
        segment = segments[i]
        name = segment.attributes['name']
        if name is not None:
            name_elems = name.split('_', 4)
            if len(name_elems) > 1 and name_elems[0] == 'ts':
                start = utils.ms_to_webvtt(int(name_elems[1]))
                end = utils.ms_to_webvtt(int(name_elems[2]))
                spkr = name_elems[3]
                txt = vtt_escape(html_node_to_text(segment))
                vtt += f'{i+1}\n{start} --> {end}\n<v {spkr}>{txt.lstrip()}\n\n'
    return vtt


def html_to_srt(parser: AdvancedHTMLParser.AdvancedHTMLParser) -> str:
    def ms_to_srt_ts(ms: int) -> str:
        s, ms = divmod(ms, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    srt = ''
    segments = parser.getElementsByTagName('a')
    for i in range(len(segments)):
        segment = segments[i]
        name = segment.attributes['name']
        if name is not None:
            name_elems = name.split('_', 4)
            if len(name_elems) > 1 and name_elems[0] == 'ts':
                start = ms_to_srt_ts(int(name_elems[1]))
                end = ms_to_srt_ts(int(name_elems[2]))
                spkr = name_elems[3]
                txt = html_node_to_text(segment).strip().replace('\n\n', '\n')
                srt += f'{i+1}\n{start} --> {end}\n[{spkr}] {txt}\n\n'
    return srt


# --- Diarization helpers ----------------------------------------------

def overlap_len(ss_start, ss_end, ts_start, ts_end):
    """Calculate overlap percentage between a speaker segment and a transcript segment.

    Returns None if ts is before ss, 0.0 if ts is after ss, else overlap ratio.
    """
    if ts_end < ss_start:
        return None
    if ts_start > ss_end:
        return 0.0
    ts_len = ts_end - ts_start
    if ts_len <= 0:
        return None
    overlap_start = max(ss_start, ts_start)
    overlap_end = min(ss_end, ts_end)
    ol_len = overlap_end - overlap_start + 1
    return ol_len / ts_len


def find_speaker(diarization, transcript_start, transcript_end,
                 speaker_name_map=None, overlapping_enabled=True) -> str:
    """Find the speaker for a transcript segment based on diarization data.

    Returns speaker label string, or empty string if no overlap found.
    Prefixes with '//' for overlapping speech.
    """
    if speaker_name_map is None:
        speaker_name_map = {}

    spkr = ''
    overlap_found = 0
    overlap_threshold = 0.8
    segment_len = 0
    is_overlapping = False

    for segment in diarization:
        t = overlap_len(segment["start"], segment["end"], transcript_start, transcript_end)
        if t is None:
            break

        current_segment_len = segment["end"] - segment["start"]
        lbl = segment["label"]
        current_segment_spkr = speaker_name_map.get(lbl, f'S{lbl[8:]}')

        if overlap_found >= overlap_threshold:
            if (t >= overlap_threshold) and (current_segment_len < segment_len):
                is_overlapping = True
                overlap_found = t
                segment_len = current_segment_len
                spkr = current_segment_spkr
        elif t > overlap_found:
            overlap_found = t
            segment_len = current_segment_len
            spkr = current_segment_spkr

    if overlapping_enabled and is_overlapping:
        return f"//{spkr}"
    else:
        return spkr
