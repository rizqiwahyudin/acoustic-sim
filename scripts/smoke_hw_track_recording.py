"""Bounded live test of adaptive tracking while hardware audio recording is active."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import time
from urllib.request import urlopen
import wave

from websockets.sync.client import connect


WS_URL = "ws://127.0.0.1:8766/realtime_hw"
WAV_URL = "http://127.0.0.1:8766/hw_recording.wav"


def receive_until(socket, predicate, limit=2000):
    for _ in range(limit):
        message = json.loads(socket.recv(timeout=10))
        if predicate(message):
            return message
    raise RuntimeError("timed out waiting for firmware state")


def send(socket, command):
    socket.send(json.dumps({"type": "command", "command": command}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scans", type=int, default=1)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    if args.scans < 1:
        raise ValueError("--scans must be at least 1")

    with connect(WS_URL, open_timeout=5) as socket:
        initial = json.loads(socket.recv(timeout=5))
        if initial.get("firmware_mode") != "IDLE":
            raise RuntimeError(f"firmware was not idle: {initial.get('firmware_mode')}")

        send(socket, "R,1")
        receive_until(socket, lambda frame: frame.get("recording_state") == "recording")

        initial_scans = initial.get("scan_count", 0)
        send(socket, "G")
        if args.seconds is not None:
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                json.loads(socket.recv(timeout=10))
        else:
            receive_until(
                socket,
                lambda frame: frame.get("scan_count", 0) >= initial_scans + args.scans,
                limit=max(2000, args.scans * 500),
            )

        send(socket, "X")
        receive_until(socket, lambda frame: frame.get("firmware_mode") == "IDLE")

        send(socket, "R,0")
        stopped = receive_until(socket, lambda frame: frame.get("recording_state") == "ready")
        recording = stopped["last_recording"]
        if recording["bytes"] == 0 or recording["overruns"] != 0:
            raise RuntimeError(f"invalid recording telemetry: {recording}")

        send(socket, "D")
        complete = receive_until(socket, lambda frame: frame.get("audio_download_ready") is True)
        if complete.get("audio_download_error"):
            raise RuntimeError(complete["audio_download_error"])

    with urlopen(WAV_URL, timeout=10) as response:
        wav_bytes = response.read()
    with wave.open(BytesIO(wav_bytes), "rb") as wav:
        summary = {
            "frames": wav.getnframes(),
            "sample_rate_hz": wav.getframerate(),
            "channels": wav.getnchannels(),
            "overruns": recording["overruns"],
            "scan_count": stopped["scan_count"],
            "protocol_errors": stopped["protocol_errors"],
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
