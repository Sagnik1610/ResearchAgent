"""Utility helpers for parsing structured LLM responses.

These helpers make the downstream agents resilient to common formatting
variations (markdown bold, headings repeated with extra whitespace, ratings
expressed as ``4/5`` etc.).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional


def _normalise_newlines(text: str) -> str:
    """Return ``text`` with Windows/old Mac newlines normalised to ``\n``."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def strip_markdown(text: Optional[str]) -> str:
    """Remove lightweight markdown decorations that break simple regex parsing."""

    if not text:
        return ""
    cleaned = _normalise_newlines(text)
    # Bold/italics markers frequently wrap section headers (e.g. **Review:**).
    return cleaned.replace("**", "").replace("__", "")


def _label_pattern(label: str) -> str:
    """Regex pattern that matches a label optionally followed by ``(…)``."""

    escaped = re.escape(label)
    return rf"{escaped}(?:\s*\([^)]*\))?"


def parse_structured_sections(text: str, labels: Iterable[str]) -> Dict[str, Optional[str]]:
    """Extract labelled sections from ``text``.

    Parameters
    ----------
    text:
        The raw LLM response.
    labels:
        The ordered labels to extract. The parser assumes each label appears at
        most once and stops a section at the start of the next label.

    Returns
    -------
    Dict[str, Optional[str]]
        Mapping from lower-cased label name to the extracted section body. Any
        missing labels are returned as ``None``.
    """

    cleaned = strip_markdown(text)
    section_map: Dict[str, Optional[str]] = {}
    ordered_labels: List[str] = list(labels)

    separator_pattern = r"(?:[:\-–—]\s*)?"

    bullet_pattern = r"(?:(?:[-*•]\s*)|(?:\d+[\.)]\s*))?"

    for index, label in enumerate(ordered_labels):
        following = ordered_labels[index + 1 :]
        label_pattern = _label_pattern(label)
        if following:
            following_pattern = "|".join(_label_pattern(next_label) for next_label in following)
            prefixed_following = rf"{bullet_pattern}(?:{following_pattern})"
            regex = (
                rf"^\s*{bullet_pattern}{label_pattern}\s*{separator_pattern}(.*?)(?=\n\s*(?:{prefixed_following})\s*{separator_pattern}|\Z)"
            )
        else:
            regex = rf"^\s*{bullet_pattern}{label_pattern}\s*{separator_pattern}(.*)"

        match = re.search(
            regex,
            cleaned,
            flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        section_map[label.lower()] = match.group(1).strip() if match else None

    return section_map


_RATING_WITH_LABEL = re.compile(
    r"Rating(?:\s*\([^)]*\))?\s*[:\-]?\s*([1-5])(?:\s*/\s*5)?",
    flags=re.IGNORECASE,
)
_RATING_SLASH_FIVE = re.compile(r"\b([1-5])\s*/\s*5\b")


def extract_numeric_rating(text: Optional[str]) -> Optional[int]:
    """Extract a 1–5 rating from ``text`` if present.

    The helper is resilient to common LLM formats such as ``Rating (1-5): 4/5``
    or ``Rating: 3`` spread across newlines. It first looks for an explicit
    ``Rating`` label to avoid confusing list numbering with scores, falling back
    to matching ``4/5`` style snippets when no label is present.
    """

    if not text:
        return None

    cleaned = strip_markdown(text)
    labelled_match = _RATING_WITH_LABEL.search(cleaned)
    if labelled_match:
        return int(labelled_match.group(1))

    slash_match = _RATING_SLASH_FIVE.search(cleaned)
    if slash_match:
        return int(slash_match.group(1))

    return None


def parse_review_feedback_rating(text: str) -> Dict[str, Optional[object]]:
    """Parse a validator response into review/feedback/rating fields."""

    sections = parse_structured_sections(text, ["Review", "Feedback", "Rating"])

    rating_value: Optional[int] = extract_numeric_rating(sections.get("rating"))
    if rating_value is None:
        rating_value = extract_numeric_rating(text)

    return {
        "review": sections.get("review"),
        "feedback": sections.get("feedback"),
        "rating": rating_value,
    }
