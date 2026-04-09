"""
utils.py
-----------
Pre-splits documents into coarse sections (H1/H2 headings or ---PAGE--- 
delimiters) before passing them to the LLM, keeping prompt sizes manageable.
"""

import re


def _split_into_sections(text: str, metadata: dict) -> list[dict]:
    """
    Splits a Markdown document into sections by H1/H2 headings, falling back
    to ---PAGE--- delimiters if no headings are found. Returns a list of dicts
    with 'content' and 'metadata' keys.
    """
    pattern = r"(?=^#{1,2} )"
    raw_secs = re.split(pattern, text, flags=re.MULTILINE)

    # Fallback to page boundaries if no headings were found
    if len(raw_secs) <= 1:
        raw_secs = text.split("---PAGE---")

    return [
        {"content": s.strip(), "metadata": dict(metadata)}
        for s in raw_secs
        if s.strip()
    ]