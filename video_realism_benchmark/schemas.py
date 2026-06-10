"""Structured-output schema and strict binary validation."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from .config import ALLOWED_BINARY_LABELS, LIGHT_SOURCE_DIRECTIONS


class BinaryDecision(str, Enum):
    """Gemini structured output enum: only lowercase yes/no are valid."""

    YES = "yes"
    NO = "no"


BINARY_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "STRING",
    "enum": list(ALLOWED_BINARY_LABELS),
}


GEMINI_LINE_SELECTION_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "OBJECT",
    "properties": {
        "usable": {"type": "BOOLEAN"},
        "frame_assessment": {"type": "STRING"},
        "selected_candidate_ids": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": ["usable", "frame_assessment", "selected_candidate_ids"],
}


LIGHT_SOURCE_ANALYSIS_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "OBJECT",
    "properties": {
        "single_light_source_consistency": {
            "type": "STRING",
            "enum": list(ALLOWED_BINARY_LABELS),
        },
        "light_sources": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "source_id": {"type": "STRING"},
                    "relative_direction_from_frame_center": {
                        "type": "STRING",
                        "enum": list(LIGHT_SOURCE_DIRECTIONS),
                    },
                },
                "required": [
                    "source_id",
                    "relative_direction_from_frame_center",
                ],
            },
        },
    },
    "required": [
        "single_light_source_consistency",
        "light_sources",
    ],
}


PROMPT_OBJECT_VISIBILITY_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "OBJECT",
    "properties": {
        "prompt_object_recognizability": {
            "type": "STRING",
            "enum": list(ALLOWED_BINARY_LABELS),
        },
        "objects_of_interest": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "object_name": {"type": "STRING"},
                    "visible_overall": {"type": "BOOLEAN"},
                    "per_frame_visibility": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "frame_index": {"type": "NUMBER"},
                                "timestamp_sec": {"type": "NUMBER"},
                                "visible": {"type": "BOOLEAN"},
                            },
                            "required": [
                                "frame_index",
                                "timestamp_sec",
                                "visible",
                            ],
                        },
                    },
                },
                "required": [
                    "object_name",
                    "visible_overall",
                    "per_frame_visibility",
                ],
            },
        },
    },
    "required": [
        "prompt_object_recognizability",
        "objects_of_interest",
    ],
}


BinaryLabel = Literal["yes", "no"]


def validate_binary_label(value: object) -> BinaryLabel:
    """Validate that a model response is exactly one allowed label."""

    if not isinstance(value, str):
        raise ValueError(f"Gemini returned a non-string binary decision: {value!r}")
    normalized = value.strip()
    if normalized not in ALLOWED_BINARY_LABELS:
        raise ValueError(
            f"Gemini returned {value!r}; expected exactly one of {ALLOWED_BINARY_LABELS}."
        )
    return normalized  # type: ignore[return-value]
