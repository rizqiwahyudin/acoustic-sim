"""Exercise hardware Record/Stop/Download through the live GUI backend."""

from __future__ import annotations

from array import array
from io import BytesIO
import json
import math
from urllib.request import urlopen
import wave

from websockets.sync.client import connect


WS_URL = "ws://127.0.0.1:8766/realtime_hw"
WAV_URL = "http://127.0.0.1:8766/hw_recording.wav"


def receive_until(socket, predicate, limit=1000):
    for _ in range(limit):
        message = json.loads(socket.recv(timeout=10))
        if predicate(message):
            return message
    raise RuntimeError("timed out waiting for hardware recording state")


def send_command(socket, command):
    socket.send(json.dumps({"type": "command", "command": command}))


def main():
    with connect(WS_URL, open_timeout=5) as socket:
        initial = json.loads(socket.recv(timeout=5))
        if initial.get("type") != "init" or not initial.get("connected"):
            raise RuntimeError("hardware transport is not connected")

        send_command(socket, "R,1")
        receive_until(socket, lambda message: message.get("recording_state") == "recording")

        initial_scan_count = initial.get("scan_count", 0)
        send_command(socket, "F")
        receive_until(
            socket,
            lambda message: message.get("scan_count", 0) > initial_scan_count,
        )

        send_command(socket, "R,0")
        stopped = receive_until(
            socket,
            lambda message: message.get("recording_state") == "ready",
        )
        recording = stopped["last_recording"]
        if recording["bytes"] == 0 or recording["overruns"] != 0:
            raise RuntimeError(f"invalid recording telemetry: {recording}")

        send_command(socket, "D")
        complete = receive_until(
            socket,
            lambda message: message.get("audio_download_ready") is True,
        )
        if complete.get("audio_download_error"):
            raise RuntimeError(complete["audio_download_error"])

    with urlopen(WAV_URL, timeout=10) as response:
        wav_bytes = response.read()
    with wave.open(BytesIO(wav_bytes), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
        samples = array("h")
        samples.frombytes(pcm)
        summary = {
            "sample_rate_hz": wav.getframerate(),
            "channels": wav.getnchannels(),
            "sample_width_bytes": wav.getsampwidth(),
            "frames": wav.getnframes(),
            "wav_bytes": len(wav_bytes),
            "minimum": min(samples),
            "maximum": max(samples),
            "rms": math.sqrt(sum(sample * sample for sample in samples) / len(samples)),
        }
    if summary["minimum"] == 0 and summary["maximum"] == 0:
        raise RuntimeError("captured WAV is silent")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
