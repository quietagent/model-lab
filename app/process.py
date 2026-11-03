"""Is a given process still running? Used to tell a live benchmark from one whose server died.

A process is identified by its ID together with the time it started: the operating system hands
IDs out again, so an ID alone could point at an unrelated newer process.
"""
from __future__ import annotations

import os
import sys


def _windows(pid: int) -> tuple[bool, float | None]:
    import ctypes
    from ctypes import wintypes

    query_limited, still_active, access_denied = 0x1000, 259, 5
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

    handle = kernel32.OpenProcess(query_limited, False, pid)
    if not handle:
        # Access denied: the process exists but belongs to someone else. Anything else: no such process.
        return ctypes.get_last_error() == access_denied, None
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != still_active:
            return False, None                     # it has exited; something still holds a handle to it
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in (created, exited, kernel, user))):
            return True, None
        ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime     # 100 ns steps since 1601
        return True, ticks / 1e7 - 11644473600.0
    finally:
        kernel32.CloseHandle(handle)


def _posix(pid: int) -> tuple[bool, float | None]:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, None
    except PermissionError:
        return True, None
    except OSError:
        return False, None
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as handle:
            # Field 22 (start time, in clock ticks since boot); the name in field 2 may contain spaces.
            return True, float(handle.read().rpartition(")")[2].split()[19])
    except (OSError, ValueError, IndexError):
        return True, None                          # no /proc here: the ID is all there is to go on


def describe(pid: int) -> tuple[bool, float | None]:
    """(is it running, when it started). The start time is None where the system cannot tell."""
    if not isinstance(pid, int) or pid <= 0:
        return False, None
    try:
        return _windows(pid) if sys.platform == "win32" else _posix(pid)
    except Exception:
        return False, None


def started(pid: int | None = None) -> float | None:
    """When a process started (this one by default), or None if unknown."""
    return describe(os.getpid() if pid is None else pid)[1]


def is_alive(pid: int | None, started_at: float | None) -> bool:
    """True if the process with this ID is running and is the same one that started at `started_at`."""
    if pid is None:
        return False
    alive, now_started = describe(int(pid))
    if not alive:
        return False
    if started_at is not None and now_started is not None and abs(now_started - started_at) > 0.01:
        return False                               # the ID was reused by a newer process
    return True
