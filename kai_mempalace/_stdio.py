"""Stdio UTF-8 reconfiguration helper for Windows entry points."""

from __future__ import annotations

import sys
from typing import Callable, Optional


def reconfigure_stdio_utf8_on_windows(
    *,
    stdin_errors: str = "surrogateescape",
    stdout_errors: str = "strict",
    stderr_errors: str = "strict",
    on_failure: Optional[Callable[[str, BaseException], None]] = None,
) -> None:
    if sys.platform != "win32":
        return

    policies = (
        ("stdin", stdin_errors),
        ("stdout", stdout_errors),
        ("stderr", stderr_errors),
    )
    for name, errors in policies:
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors=errors)
        except Exception as exc:
            if on_failure is not None:
                on_failure(name, exc)
            else:
                print(
                    f"WARNING: Could not reconfigure {name} to UTF-8: {exc}",
                    file=sys.stderr,
                )
