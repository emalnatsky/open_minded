"""Print the microphones Python/PyAudio can see for CRI diagnostics."""

from __future__ import annotations

import sys

from audio_devices import AudioDeviceError, format_input_device_list


def main() -> int:
    try:
        print(format_input_device_list())
        return 0
    except AudioDeviceError as exc:
        print(f"Could not list audio devices: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
