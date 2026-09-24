from __future__ import annotations

import io
import json
import time
import wave

from fastapi.testclient import TestClient
import numpy as np

import sim_server
from heimdall_acoustic import ACOUSTIC_MODEL_VERSION, AcousticCache, PreparationCancelled


class RecordingTransport:
    def __init__(self, fail_stop_command=False):
        self.actions = []
        self.fail_stop_command = fail_stop_command

    def send_command(self, command):
        self.actions.append(("command", command))
        if self.fail_stop_command:
            raise OSError("link unavailable")

    def stop(self):
        self.actions.append(("stop", None))


def test_transport_replacement_stops_previous_transport_first():
    previous = RecordingTransport()
    sim_server._heimdall_transport = previous

    sim_server._replace_heimdall_transport(None)

    assert previous.actions == [("command", "X"), ("stop", None)]


def test_transport_replacement_closes_after_stop_command_failure():
    previous = RecordingTransport(fail_stop_command=True)
    sim_server._heimdall_transport = previous

    sim_server._replace_heimdall_transport(None)

    assert previous.actions == [("command", "X"), ("stop", None)]


def test_hardware_recording_wav_endpoint_packages_pcm():
    class AudioTransport(RecordingTransport):
        def get_audio_download(self):
            return {
                "sample_rate_hz": 48000,
                "bits_per_sample": 16,
                "channels": 1,
                "pcm": b"\x01\x00\xff\xff\x02\x00\xfe\xff",
            }

    transport = AudioTransport()
    sim_server._heimdall_transport = transport
    with TestClient(sim_server.app) as client:
        response = client.get("/hw_recording.wav")
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        with wave.open(io.BytesIO(response.content), "rb") as wav:
            assert wav.getframerate() == 48000
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getnframes() == 4


def test_emulator_connect_command_stream_and_disconnect():
    with TestClient(sim_server.app) as client:
        response = client.post(
            "/hw_connect",
            json={
                "transport": "emulator",
                "rows": 3,
                "columns": 4,
                "azimuth_min_deg": -40,
                "azimuth_max_deg": 40,
                "elevation_min_deg": -30,
                "elevation_max_deg": 30,
                "flight_profile": "orbit",
                "flight_speed": 1.5,
            },
        )
        assert response.status_code == 200

        status = client.get("/hw_status").json()
        assert status["available"]
        assert status["transport"] == "emulator"

        with client.websocket_connect("/realtime_hw") as websocket:
            initial = websocket.receive_json()
            assert initial["type"] == "init"

            websocket.send_json({"type": "command", "command": "F"})
            complete = None
            for _ in range(100):
                frame = websocket.receive_json()
                if frame["type"] == "frame" and frame["scan_count"] >= 1:
                    complete = frame
                    break

            assert complete is not None
            assert complete["configuration"] == {
                "rows": 3,
                "columns": 4,
                "sectors": 12,
                "microphones": 44,
            }
            assert len(complete["azimuth_deg"]) == 4
            assert len(complete["elevation_deg"]) == 3
            assert complete["azimuth_deg"] == [-30.0, -10.0, 10.0, 30.0]
            assert complete["elevation_deg"] == [-20.0, 0.0, 20.0]
            assert all(level is not None for row in complete["levels_raw"] for level in row)
            assert "target_update_rate_hz" in complete
            assert "target_update_age_s" in complete
            assert complete["emulator_flight_profile"] == "orbit"
            assert complete["emulator_flight_speed"] == 1.5
            assert complete["grid_source"] == "fast_emulator"
            assert complete["emulator_truth"] is None

        response = client.post("/hw_disconnect")
        assert response.status_code == 200
        assert not client.get("/hw_status").json()["available"]


def test_acoustic_prepare_connect_and_stream(monkeypatch, tmp_path):
    levels_path = tmp_path / "levels_raw.npy"
    np.save(levels_path, np.full((4, 36), 1000, dtype=np.uint32))
    manifest = {
        "schema": ACOUSTIC_MODEL_VERSION,
        "cache_id": "acoustic-test",
        "scenario": {"scenario": "conference_evasive"},
        "level_step_s": 0.005,
        "preparation_seconds": 0.01,
        "calibrated": False,
    }
    cache = AcousticCache(
        "acoustic-test", tmp_path, manifest, np.load(levels_path, mmap_mode="r")
    )
    monkeypatch.setattr(sim_server, "prepare_acoustic_cache", lambda *args, **kwargs: cache)
    manager = sim_server.AcousticPreparationManager()
    monkeypatch.setattr(sim_server, "_acoustic_preparation", manager)

    with TestClient(sim_server.app) as client:
        response = client.post("/hw_acoustic_prepare", json={"scenario": "conference_evasive"})
        assert response.status_code == 200
        deadline = time.monotonic() + 1.0
        while manager.snapshot()["state"] != "ready" and time.monotonic() < deadline:
            time.sleep(0.005)
        assert manager.snapshot()["cache_id"] == "acoustic-test"

        response = client.post(
            "/hw_connect",
            json={"transport": "acoustic", "acoustic_cache_id": "acoustic-test"},
        )
        assert response.status_code == 200
        with client.websocket_connect("/realtime_hw") as websocket:
            initial = websocket.receive_json()
            assert initial["transport"] == "acoustic-emulator"
            assert initial["acoustic_cache_id"] == "acoustic-test"
            assert initial["acoustic_scenario"] == "conference_evasive"
            assert not initial["acoustic_calibrated"]
            assert initial["grid_source"] == "deployment_contract"
            assert initial["emulator_truth"]["range_m"] > 0.0
            assert len(initial["emulator_truth"]["room_position_m"]) == 3
        client.post("/hw_disconnect")


def test_custom_acoustic_grid_connects_without_changing_serial_contract(monkeypatch, tmp_path):
    levels_path = tmp_path / "levels_raw.npy"
    np.save(levels_path, np.full((4, 99), 1000, dtype=np.uint32))
    grid = {
        "rows": 9,
        "columns": 11,
        "azimuth_min_deg": -80.0,
        "azimuth_max_deg": 80.0,
        "elevation_min_deg": -60.0,
        "elevation_max_deg": 60.0,
    }
    cache = AcousticCache("custom-grid", tmp_path, {
        "schema": ACOUSTIC_MODEL_VERSION,
        "cache_id": "custom-grid",
        "scenario": {"scenario": "handheld_2m_drone"},
        "grid": grid,
        "grid_mode": "exploratory_simulation",
        "level_step_s": 0.005,
        "preparation_seconds": 0.01,
        "calibrated": False,
    }, np.load(levels_path, mmap_mode="r"))
    manager = sim_server.AcousticPreparationManager()
    manager.cache = cache
    monkeypatch.setattr(sim_server, "_acoustic_preparation", manager)

    with TestClient(sim_server.app) as client:
        response = client.post(
            "/hw_connect",
            json={"transport": "acoustic", "acoustic_cache_id": "custom-grid"},
        )
        assert response.status_code == 200
        with client.websocket_connect("/realtime_hw") as websocket:
            initial = websocket.receive_json()
            assert initial["configuration"]["rows"] == 9
            assert initial["configuration"]["columns"] == 11
            assert initial["configuration"]["sectors"] == 99
            assert initial["grid_source"] == "exploratory_simulation"
        client.post("/hw_disconnect")


def test_acoustic_audition_endpoint_returns_all_jointly_normalized_clips(monkeypatch, tmp_path):
    levels_path = tmp_path / "levels_raw.npy"
    np.save(levels_path, np.ones((4, 49), dtype=np.uint32))
    cache = AcousticCache("audition", tmp_path, {
        "schema": ACOUSTIC_MODEL_VERSION,
        "cache_id": "audition",
        "scenario": {"scenario": "stationary_2m"},
        "level_step_s": 0.005,
        "preparation_seconds": 0.01,
        "calibrated": False,
    }, np.load(levels_path, mmap_mode="r"))
    manager = sim_server.AcousticPreparationManager()
    manager.cache = cache
    monkeypatch.setattr(sim_server, "_acoustic_preparation", manager)
    monkeypatch.setattr(sim_server, "render_acoustic_audition", lambda *_args, **_kwargs: {
        "sample_rate_hz": 48000,
        "start_s": 1.0,
        "duration_s": 0.25,
        "selected_sector": 24,
        "truth_azimuth_deg": 0.0,
        "truth_elevation_deg": 0.0,
        "clips": {name: np.ones(12000) * (index + 1) for index, name in enumerate(
            ("generated", "single_mic", "unsteered", "truth", "selected")
        )},
    })

    with TestClient(sim_server.app) as client:
        response = client.post("/hw_acoustic_audition", json={
            "acoustic_cache_id": "audition",
            "start_s": 1.0,
            "duration_s": 0.25,
            "selected_sector": 24,
        })
        assert response.status_code == 200
        payload = response.json()
        assert set(payload["clips_b64"]) == {
            "generated", "single_mic", "unsteered", "truth", "selected",
        }
        assert payload["normalization"] == "joint_peak"


def test_acoustic_prepare_cancel(monkeypatch):
    def wait_for_cancel(*_args, cancel_event, **_kwargs):
        if not cancel_event.wait(timeout=1.0):
            raise AssertionError("cancel signal was not received")
        raise PreparationCancelled("canceled")

    monkeypatch.setattr(sim_server, "prepare_acoustic_cache", wait_for_cancel)
    manager = sim_server.AcousticPreparationManager()
    monkeypatch.setattr(sim_server, "_acoustic_preparation", manager)
    with TestClient(sim_server.app) as client:
        assert client.post("/hw_acoustic_prepare", json={"scenario": "stationary_2m"}).status_code == 200
        assert client.post("/hw_acoustic_cancel").status_code == 200
        deadline = time.monotonic() + 1.0
        while manager.snapshot()["state"] == "preparing" and time.monotonic() < deadline:
            time.sleep(0.005)
        assert manager.snapshot()["state"] == "canceled"