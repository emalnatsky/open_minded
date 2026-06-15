"""Terminal helpers for a clean researcher-facing CRI session."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def terminal_log_level(default: int = logging.WARNING) -> int:
    if env_flag("CRI_VERBOSE_LOGS", False):
        return logging.INFO
    raw = os.environ.get("CRI_TERMINAL_LOG_LEVEL", "").strip().upper()
    if not raw:
        return default
    return getattr(logging, raw, default)


def clear_terminal_status_line(stream=None) -> None:
    stream = stream or sys.stdout
    try:
        width = 96
        stream.write("\r" + (" " * width) + "\r")
        stream.flush()
    except Exception:
        pass


class LoadingStatus:
    """A tiny cross-platform one-line loading status."""

    _active = None
    _lock = threading.Lock()

    def __init__(self, label: str = "Loading", enabled: bool = True, stream=None):
        self.label = label
        self.enabled = enabled
        self.stream = stream or sys.stdout
        self._stop = threading.Event()
        self._thread = None
        self._started = 0.0
        self._last_text = ""
        self._stopped = False
        self._needs_leading_blank = True

    def __enter__(self):
        if not self.enabled:
            return self
        self._started = time.monotonic()
        with self._lock:
            LoadingStatus._active = self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop(keep_line=(exc_type is None))
        return False

    def _format_status(self) -> str:
        elapsed = int(time.monotonic() - self._started)
        minutes, seconds = divmod(elapsed, 60)
        if self.label:
            return f"{self.label}... {minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _run(self):
        while not self._stop.is_set():
            text = self._format_status()
            try:
                clear_terminal_status_line(self.stream)
                if self._needs_leading_blank:
                    self.stream.write("\n")
                    self._needs_leading_blank = False
                self._last_text = text
                self.stream.write(text)
                self.stream.flush()
            except Exception:
                return
            self._stop.wait(0.35)

    def stop(self, keep_line: bool = False):
        if not self.enabled or self._stopped:
            return
        self._stopped = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with self._lock:
            if LoadingStatus._active is self:
                LoadingStatus._active = None
        if keep_line:
            try:
                text = self._format_status()
                clear_terminal_status_line(self.stream)
                self.stream.write(text + "\n\n")
                self.stream.flush()
            except Exception:
                pass
        else:
            clear_terminal_status_line(self.stream)

    @classmethod
    def clear_active(cls):
        cls.stop_active(keep_line=False)

    @classmethod
    def clear_active_line(cls):
        with cls._lock:
            active = cls._active
        if active is not None and active.enabled and not active._stopped:
            clear_terminal_status_line(active.stream)
            active._needs_leading_blank = True

    @classmethod
    def stop_active(cls, keep_line: bool = False):
        with cls._lock:
            active = cls._active
        if active is not None:
            active.stop(keep_line=keep_line)


class ClearLoadingBeforeWarningFilter(logging.Filter):
    """Clear the loading line before warnings/errors are emitted."""

    def filter(self, record):
        if record.levelno >= logging.WARNING:
            LoadingStatus.clear_active_line()
        return True


def install_warning_line_clear_filter() -> ClearLoadingBeforeWarningFilter:
    filter_obj = ClearLoadingBeforeWarningFilter()
    root_logger = logging.getLogger()
    if not any(isinstance(existing, ClearLoadingBeforeWarningFilter) for existing in root_logger.filters):
        root_logger.addFilter(filter_obj)
    for handler in root_logger.handlers:
        if not any(isinstance(existing, ClearLoadingBeforeWarningFilter) for existing in handler.filters):
            handler.addFilter(filter_obj)
    if not getattr(logging, "_cri_loading_record_factory_installed", False):
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            if record.levelno >= logging.WARNING:
                LoadingStatus.clear_active_line()
            return record

        logging.setLogRecordFactory(record_factory)
        logging._cri_loading_record_factory_installed = True
    return filter_obj
