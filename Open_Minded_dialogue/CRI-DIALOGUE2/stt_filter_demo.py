"""Show how the CRI STT quality filter behaves in different scopes.

This demo does not run Whisper. It measures only the post-transcription text
check, so the timing shows the extra latency added by the safety filter.
"""

from __future__ import annotations

import time

from speech_io import SpeechIO


CASES = (
    ("I like hockey because it is fast", "casual"),
    ("computer games", "casual"),
    ("oui je ne sais pas", "memory-risk"),
    ("porque no quiero", "memory-risk"),
    ("pizza", "memory-risk"),
    ("qwrty psstt xkcd zzzpq", "memory-risk"),
)


def _format_decision(decision: dict) -> str:
    return decision["result"]


def build_rows() -> list[dict]:
    speech = SpeechIO(use_keyboard_input_fn=lambda: True)
    rows = []
    for transcript, turn_type in CASES:
        memory_risk = turn_type == "memory-risk"
        start = time.perf_counter()
        reason = speech.stt_quality_rejection_reason(transcript)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        rows.append(
            {
                "transcript": transcript,
                "turn": turn_type,
                "off": _format_decision(
                    speech.stt_quality_filter_decision(
                        transcript,
                        memory_risk=memory_risk,
                        scope="memory",
                        enabled=False,
                    )
                ),
                "memory": _format_decision(
                    speech.stt_quality_filter_decision(
                        transcript,
                        memory_risk=memory_risk,
                        scope="memory",
                        enabled=True,
                    )
                ),
                "all": _format_decision(
                    speech.stt_quality_filter_decision(
                        transcript,
                        memory_risk=memory_risk,
                        scope="all",
                        enabled=True,
                    )
                ),
                "ms": f"{elapsed_ms:.3f}",
                "reason": reason or "-",
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    columns = (
        ("transcript", "transcript"),
        ("turn", "turn type"),
        ("off", "filter off"),
        ("memory", "memory scope"),
        ("all", "all scope"),
        ("ms", "check ms"),
    )
    widths = {
        key: max(len(label), *(len(str(row[key])) for row in rows))
        for key, label in columns
    }
    header = " | ".join(label.ljust(widths[key]) for key, label in columns)
    separator = "-+-".join("-" * widths[key] for key, _label in columns)
    print(header)
    print(separator)
    for row in rows:
        print(" | ".join(str(row[key]).ljust(widths[key]) for key, _label in columns))


def main() -> int:
    print("STT quality filter demo")
    print("Shared default after git pull: CRI_STT_QUALITY_FILTER_SCOPE=all")
    print()
    print_table(build_rows())
    print()
    print(
        "In all scope, suspicious transcripts retry for every child answer. "
        "Use CRI_STT_QUALITY_FILTER=false locally to turn it off during testing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
