"""Retry downloading the finalized hardware recording already stored in APS6404."""

from __future__ import annotations

from io import BytesIO
import json
from urllib.request import urlopen
import wave

from websockets.sync.client import connect


with connect("ws://127.0.0.1:8766/realtime_hw", open_timeout=5) as socket:
    initial = json.loads(socket.recv(timeout=5))
    if initial.get("firmware_mode") != "IDLE":
        raise RuntimeError(f"firmware must be idle, got {initial.get('firmware_mode')}")
    socket.send(json.dumps({"type": "command", "command": "D"}))
    for _ in range(1000):
        frame = json.loads(socket.recv(timeout=45))
        if frame.get("audio_download_error"):
            raise RuntimeError(frame["audio_download_error"])
        if frame.get("audio_download_ready"):
            break
    else:
        raise RuntimeError("audio download did not complete")

with urlopen("http://127.0.0.1:8766/hw_recording.wav", timeout=10) as response:
    wav_bytes = response.read()
with wave.open(BytesIO(wav_bytes), "rb") as wav:
    print(json.dumps({
        "frames": wav.getnframes(),
        "sample_rate_hz": wav.getframerate(),
        "channels": wav.getnchannels(),
        "wav_bytes": len(wav_bytes),
    }, indent=2))
