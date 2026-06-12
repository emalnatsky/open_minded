"""Small PyAudio helpers for CRI microphone diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


class AudioDeviceError(RuntimeError):
    """Raised when an explicit microphone selection cannot be used."""


@dataclass(frozen=True)
class AudioInputDevice:
    index: int
    name: str
    channels: int
    sample_rate: float
    is_default: bool = False


def _clean_device_name(name) -> str:
    text = str(name or "<unnamed input>")
    return " ".join(text.split())


def parse_optional_mic_index(value) -> int | None:
    """Return an explicit mic index, or None for the system/default mic."""
    raw = "" if value is None else str(value).strip()
    if raw.lower() in {"", "auto", "default", "system"}:
        return None
    try:
        index = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid CRI_STT_MIC_INDEX value: {raw!r}. "
            "Use a non-negative integer, or leave it empty for the system default mic."
        ) from exc
    if index < 0:
        raise ValueError(
            f"Invalid CRI_STT_MIC_INDEX value: {raw!r}. "
            "Use a non-negative integer, or leave it empty for the system default mic."
        )
    return index


def _load_pyaudio():
    try:
        import pyaudio
    except Exception as exc:  # pragma: no cover - depends on local install
        raise AudioDeviceError(f"PyAudio is not available: {exc}") from exc
    return pyaudio


def list_input_devices() -> list[AudioInputDevice]:
    """List PyAudio input devices and mark the current system default."""
    pyaudio = _load_pyaudio()
    pa = pyaudio.PyAudio()
    try:
        default_index = None
        try:
            default_index = int(pa.get_default_input_device_info().get("index"))
        except Exception:
            default_index = None

        devices: list[AudioInputDevice] = []
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            channels = int(info.get("maxInputChannels", 0) or 0)
            if channels <= 0:
                continue
            devices.append(
                AudioInputDevice(
                    index=index,
                    name=_clean_device_name(info.get("name")),
                    channels=channels,
                    sample_rate=float(info.get("defaultSampleRate", 0.0) or 0.0),
                    is_default=(index == default_index),
                )
            )
        return devices
    finally:
        pa.terminate()


def describe_selected_input_device(mic_index: int | None = None) -> str:
    """
    Return a researcher-facing description of the mic CRI will use.

    None means RealtimeSTT receives no input_device_index and uses the
    operating system / PortAudio default.
    """
    try:
        devices = list_input_devices()
    except AudioDeviceError:
        if mic_index is None:
            return "system default (PyAudio device list unavailable)"
        raise

    if mic_index is None:
        default = next((device for device in devices if device.is_default), None)
        if default is None:
            return "system default (PyAudio did not report a default input device)"
        return f"system default - {default.name} (index {default.index})"

    selected = next((device for device in devices if device.index == mic_index), None)
    if selected is None:
        available = ", ".join(str(device.index) for device in devices) or "none"
        raise AudioDeviceError(
            f"CRI_STT_MIC_INDEX={mic_index} is not an available input device. "
            f"Available input indexes: {available}."
        )
    return f"index {selected.index} - {selected.name}"


def format_input_device_list() -> str:
    """Format the current PyAudio input devices for terminal diagnostics."""
    devices = list_input_devices()
    lines = ["Python/PyAudio input devices:"]
    if not devices:
        lines.append("  No input devices found.")
    for device in devices:
        default = "  [system default]" if device.is_default else ""
        lines.append(
            f"  {device.index:>2}: {device.name} "
            f"channels={device.channels} rate={device.sample_rate:.0f}{default}"
        )
    lines.append("")
    lines.append("Default CRI behavior: leave CRI_STT_MIC_INDEX empty to use the system mic.")
    lines.append("Lab override only: set CRI_STT_MIC_INDEX=<index> if Windows picks the wrong mic.")
    return "\n".join(lines)
