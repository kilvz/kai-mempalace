"""Reference implementations of the reserved content transformations (RFC 002 §1.4)."""

from __future__ import annotations

import re
from typing import Protocol, Union


class Transformation(Protocol):
    def __call__(self, data: Union[bytes, str], /) -> str: ...


def utf8_replace_invalid(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def newline_normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def whitespace_trim(text: str) -> str:
    return text.strip()


_RUN_OF_THREE_OR_MORE_BLANK = re.compile(r"(?:\n[ \t]*){3,}\n")


def whitespace_collapse_internal(text: str) -> str:
    lines = text.split("\n")
    normalised = "\n".join(line if line.strip() else "" for line in lines)
    return _RUN_OF_THREE_OR_MORE_BLANK.sub("\n\n\n", normalised)


def line_trim(text: str) -> str:
    return "\n".join(line.strip() for line in text.split("\n"))


def line_join_spaces(text: str) -> str:
    paragraphs = re.split(r"\n[ \t]*\n", text)
    joined = [" ".join(line.strip() for line in p.split("\n") if line.strip()) for p in paragraphs]
    return "\n\n".join(joined)


def blank_line_drop(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if line.strip())


def strip_tool_chrome(text: str) -> str:
    return text


def tool_result_truncate(text: str) -> str:
    return text


def tool_result_omitted(text: str) -> str:
    return text


def spellcheck_user(text: str) -> str:
    return text


def synthesized_marker(text: str) -> str:
    return text


def speaker_role_assignment(text: str) -> str:
    return text


RESERVED_TRANSFORMATIONS: dict[str, Transformation] = {
    "utf8_replace_invalid": utf8_replace_invalid,
    "newline_normalize": newline_normalize,
    "whitespace_trim": whitespace_trim,
    "whitespace_collapse_internal": whitespace_collapse_internal,
    "line_trim": line_trim,
    "line_join_spaces": line_join_spaces,
    "blank_line_drop": blank_line_drop,
    "strip_tool_chrome": strip_tool_chrome,
    "tool_result_truncate": tool_result_truncate,
    "tool_result_omitted": tool_result_omitted,
    "spellcheck_user": spellcheck_user,
    "synthesized_marker": synthesized_marker,
    "speaker_role_assignment": speaker_role_assignment,
}


def get_transformation(name: str) -> Transformation:
    try:
        return RESERVED_TRANSFORMATIONS[name]
    except KeyError as e:
        raise KeyError(
            f"unknown transformation {name!r}; reserved names: {sorted(RESERVED_TRANSFORMATIONS)}"
        ) from e
