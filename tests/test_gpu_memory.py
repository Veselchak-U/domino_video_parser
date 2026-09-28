import subprocess

import pytest

from domino_video.gpu_memory import parse_memory, probe_memory

IDENTITY = (0x10DE, 0x1C02, 0x11C210DE)
ROW = "0x1C0210DE, 0x11C210DE, 3072, 2048\n"


@pytest.mark.parametrize(
    "text,expected",
    [
        (ROW, 2048),
        (ROW + ROW, None),
        ("", None),
        (ROW.replace("2048", "N/A"), None),
        (ROW.replace("2048", "4000"), None),
        (ROW.replace("2048", "-1"), None),
        (ROW.replace("3072", "0"), None),
        (ROW.replace("1C02", "1C03"), None),
        ("bad data", None),
    ],
)
def test_memory_values_and_adapter_identity(text, expected):
    assert parse_memory(text, IDENTITY).free_mib == expected


def test_non_nvidia_and_multiple_distinct_devices():
    assert parse_memory(ROW, (0x8086, 1, 1)).free_mib is None
    other = ROW.replace("1C02", "1C03").replace("2048", "123")
    assert parse_memory(other + ROW, IDENTITY).free_mib == 2048


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError(),
        subprocess.TimeoutExpired("smi", 3),
        subprocess.CalledProcessError(1, "smi"),
    ],
)
def test_unavailable_telemetry_is_unknown(monkeypatch, failure):
    monkeypatch.setattr("domino_video.gpu_memory.sys.platform", "win32")
    monkeypatch.setattr("domino_video.gpu_memory.adapter_identity", lambda: IDENTITY)

    def run(command, **kwargs):
        assert kwargs["timeout"] == 3
        assert kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW
        assert "--format=csv,noheader,nounits" in command
        raise failure

    monkeypatch.setattr("domino_video.gpu_memory.subprocess.run", run)
    assert probe_memory().free_mib is None
