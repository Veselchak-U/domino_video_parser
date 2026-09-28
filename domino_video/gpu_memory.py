"""Conservative memory telemetry for the selected DirectML adapter."""

import csv
import ctypes
import io
import subprocess
import sys
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class MemorySnapshot:
    free_mib: int | None
    reason: str = ""


def adapter_identity():
    class Desc(ctypes.Structure):
        _fields_ = [
            ("description", ctypes.c_wchar * 128),
            ("vendor", ctypes.c_uint),
            ("device", ctypes.c_uint),
            ("subsystem", ctypes.c_uint),
            ("revision", ctypes.c_uint),
            ("video", ctypes.c_size_t),
            ("system", ctypes.c_size_t),
            ("shared", ctypes.c_size_t),
            ("luid", ctypes.c_longlong),
        ]

    def method(pointer, index, restype, *args):
        table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *args)(table[index])

    def check(code):
        if code < 0:
            raise OSError(f"DXGI HRESULT {code:#x}")

    factory, adapter = ctypes.c_void_p(), ctypes.c_void_p()
    iid = (ctypes.c_byte * 16).from_buffer_copy(
        uuid.UUID("7b7166ec-21c7-44ae-b21a-c9ae321ae369").bytes_le
    )
    create = ctypes.WinDLL("dxgi").CreateDXGIFactory
    create.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_long
    try:
        check(create(ctypes.byref(iid), ctypes.byref(factory)))
        check(
            method(factory, 7, ctypes.c_long, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(
                factory, 0, ctypes.byref(adapter)
            )
        )
        desc = Desc()
        check(method(adapter, 8, ctypes.c_long, ctypes.POINTER(Desc))(adapter, ctypes.byref(desc)))
        return desc.vendor, desc.device, desc.subsystem
    finally:
        for pointer in (adapter, factory):
            if pointer.value:
                method(pointer, 2, ctypes.c_ulong)(pointer)


def parse_memory(text, identity):
    vendor, device, subsystem = identity
    if vendor != 0x10DE:
        return MemorySnapshot(None, "телеметрия памяти поддерживается только для NVIDIA")
    matches = []
    try:
        for row in csv.reader(io.StringIO(text)):
            pci, sub, total, free = (part.strip() for part in row)
            combined = int(pci, 16)
            if (combined & 0xFFFF, combined >> 16, int(sub, 16)) == identity:
                total, free = int(total), int(free)
                if total <= 0 or not 0 <= free <= total:
                    raise ValueError("неверный объём памяти")
                matches.append(free)
    except (ValueError, TypeError):
        return MemorySnapshot(None, "некорректные данные памяти GPU")
    if len(matches) != 1:
        return MemorySnapshot(None, "не удалось однозначно сопоставить адаптер GPU")
    return MemorySnapshot(matches[0])


def probe_memory():
    if sys.platform != "win32":
        return MemorySnapshot(None, "телеметрия доступна только на Windows")
    try:
        identity = adapter_identity()
        if identity[0] != 0x10DE:
            return MemorySnapshot(None, "телеметрия памяти поддерживается только для NVIDIA")
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=pci.device_id,pci.sub_device_id,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return parse_memory(result.stdout, identity)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return MemorySnapshot(None, f"свободная видеопамять неизвестна: {error}")
