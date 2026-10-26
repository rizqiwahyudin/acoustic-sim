from __future__ import annotations

import csv
import datetime
import io
import json
import time
import wave

from fastapi.testclient import TestClient
import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

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
                "channels": 2,
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
            assert wav.getnchannels() == 2
            assert wav.getsampwidth() == 2
            assert wav.getnframes() == 2


def test_hardware_recording_metadata_describes_channels_and_settings():
    class AudioTransport(RecordingTransport):
        transport_name = "serial:COM7"

        def get_audio_download(self):
            return {"sample_rate_hz": 48000, "bits_per_sample": 16, "channels": 2,
                    "pcm": b"\0" * 192000}

        def snapshot(self):
            return {
                "transport": self.transport_name,
                "configuration": {"rows": 7, "columns": 7, "sectors": 49, "microphones": 44},
                "map_hash": "0123456789abcdef",
                "settings": {"measure": 1, "window_us": 5000},
                "recording_meta": {"sector": 24, "ch1_source": 2, "ch1_index": 12},
                "last_steer": {"sector": 24, "azimuth_deg": 0.0, "elevation_deg": 0.0},
                "direction_set": None,
            }

    sim_server._heimdall_transport = AudioTransport()
    with TestClient(sim_server.app) as client:
        response = client.get("/hw_recording_metadata.json")
        assert response.status_code == 200
        metadata = response.json()
        assert metadata["schema"] == "heimdall-recording-v2"
        assert metadata["audio"]["channels"] == 2
        assert metadata["audio"]["duration_s"] == 1.0
        assert metadata["channel_roles"] == ["steered_beam", "bank_beam_12"]
        assert metadata["settings"]["window_us"] == 5000
        assert metadata["map_hash"] == "0123456789abcdef"
        assert metadata["recording"]["started_at"] is None
        assert metadata["label"] is None
        assert not metadata["dsp"]["checked"]


class DatasetTransport(RecordingTransport):
    transport_name = "serial:COM7"
    started = 1792998612.25

    def get_audio_download(self):
        return {"sample_rate_hz": 48000, "bits_per_sample": 16, "channels": 2, "pcm": b"\1\0\2\0" * 4800,
                "started_unix_s": self.started, "stopped_unix_s": self.started + 0.1,
                "downloaded_unix_s": self.started + 2.0}

    def snapshot(self):
        return {"transport": self.transport_name, "map_hash": "0123456789abcdef",
                "settings": {"measure": 1, "window_us": 5000, "band_low_hz": 2000},
                "recording_meta": {"sector": 24, "ch1_source": 1, "ch1_index": 3},
                "last_steer": {"sector": 24, "azimuth_deg": 10.0, "elevation_deg": -5.0}}


def test_recording_save_writes_the_labelled_record_into_the_dataset(monkeypatch, tmp_path):
    service = sim_server._dsp_service
    writable = [item for item in service.registry()["parameters"] if item["writable"]]
    multi = next(item for item in writable if item["words"] > 1)
    single = next(item for item in writable if item["words"] == 1)
    edited = multi["address"] + 1
    memory = {item["address"] + offset: word for item in writable for offset, word in enumerate(item["default_words"])}
    memory[edited] = multi["default_words"][1] + 7
    monkeypatch.setenv("HEIMDALL_DATASET_DIR", str(tmp_path))
    monkeypatch.setattr(service, "supports", lambda transport: (True, ""))
    monkeypatch.setattr(service, "memory", lambda transport: dict(memory))
    monkeypatch.setattr(service, "lost", lambda: [{"address": single["address"], "word": 99}])
    monkeypatch.setattr(sim_server, "_git_state", lambda path: {"commit": "abc", "dirty": False})
    sim_server._heimdall_transport = DatasetTransport()
    label = {"class": "drone", "source": "YouTube clip", "distance_m": 1.5, "azimuth_deg": 10, "room": "living room"}
    copy = {"suffix": "band-0.5-3.5kHz", "preset": "mcu", "kind": "band", "low_hz": 500, "high_hz": 3500}
    with TestClient(sim_server.app) as client:
        response = client.post("/hw_recording/save", json={"label": label, "listen_copy": copy})
        assert response.status_code == 200, response.text
        record = response.json()
        relabelled = client.post("/hw_recording/save", json={"label": {**label, "class": "not_drone"}})
        assert relabelled.status_code == 200, relabelled.text
    sim_server._heimdall_transport = None

    started = datetime.datetime.fromisoformat(record["recording"]["started_at"])
    assert started.tzinfo is not None
    assert started.timestamp() == DatasetTransport.started
    assert datetime.datetime.fromisoformat(record["recording"]["downloaded_at"]).timestamp() == DatasetTransport.started + 2
    assert record["label"]["class"] == "drone"
    assert record["label"]["distance_m"] == 1.5
    assert record["settings_changed"] == {"band_low_hz": 2000}
    assert record["dsp"]["edits"] == [{"block": multi["block"], "parameter": multi["name"], "index": 1,
                                        "address": edited, "word": multi["default_words"][1] + 7,
                                        "flashed": multi["default_words"][1]}]
    assert record["dsp"]["put_back_by_I"] == [{"block": single["block"], "parameter": single["name"], "index": None,
                                                "address": single["address"], "word": 99,
                                                "flashed": single["default_words"][0]}]
    stem = record["files"]["wav"].removesuffix(".wav")
    assert record["listen_copy"]["file"] == stem.split("/")[1] + "-band-0.5-3.5kHz.wav"
    assert record["host"]["git"]["project"] == {"commit": "abc", "dirty": False}
    with wave.open(str(tmp_path / record["files"]["wav"]), "rb") as wav:
        assert wav.getnchannels() == 2
        assert wav.getnframes() == 4800
    saved = json.loads((tmp_path / record["files"]["json"]).read_text(encoding="utf-8"))
    assert saved["label"]["class"] == "not_drone"
    assert saved["listen_copy"] is None
    rows = list(csv.DictReader((tmp_path / "index.csv").open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["class"] == "not_drone"
    assert rows[0]["channel_roles"] == "steered_beam;microphone_3"
    assert rows[0]["dsp_edits"] == "1"


def test_recording_save_refuses_unknown_labels_and_unsafe_copy_names(monkeypatch, tmp_path):
    monkeypatch.setenv("HEIMDALL_DATASET_DIR", str(tmp_path))
    sim_server._heimdall_transport = DatasetTransport()
    copy = {"preset": "mcu", "kind": "band"}
    with TestClient(sim_server.app) as client:
        assert client.post("/hw_recording/save", json={"label": {"class": "bird"}}).status_code == 422
        assert client.post("/hw_recording/save", json={"label": {"weather": "rain"}}).status_code == 422
        assert client.post("/hw_recording/save", json={"listen_copy": {**copy, "suffix": "../x"}}).status_code == 422
    sim_server._heimdall_transport = None
    assert not any(tmp_path.iterdir())


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


def test_only_pages_on_this_computer_reach_the_backend():
    foreign = {"Origin": "https://example.com"}
    with TestClient(sim_server.app) as client:
        assert client.post("/hw_connect", json={"transport": "emulator"}).status_code == 200
        try:
            assert client.get("/hw_status", headers=foreign).status_code == 403
            assert client.post("/hw_disconnect", headers=foreign).status_code == 403
            assert client.get("/hw_status", headers={"Origin": "http://localhost.example.com"}).status_code == 403
            local = client.get("/hw_status", headers={"Origin": "http://localhost:5173"})
            assert local.status_code == 200
            assert local.headers["access-control-allow-origin"] == "http://localhost:5173"
            with pytest.raises(WebSocketDisconnect) as refused:
                with client.websocket_connect("/realtime_hw", headers=foreign) as websocket:
                    websocket.receive_json()
            assert refused.value.code == 1008
            with client.websocket_connect("/realtime_hw", headers={"Origin": "http://127.0.0.1:8080"}) as websocket:
                assert websocket.receive_json()["type"] == "init"
            with client.websocket_connect("/realtime_hw") as websocket:
                assert websocket.receive_json()["type"] == "init"
        finally:
            client.post("/hw_disconnect")


def test_acoustic_prepare_connect_and_stream(monkeypatch, tmp_path):
    levels_path = tmp_path / "levels_raw.npy"
    np.save(levels_path, np.full((4, 49), 1000, dtype=np.uint32))
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

def test_serial_port_listing_is_read_only_and_well_formed():
    with TestClient(sim_server.app) as client:
        response = client.get("/hw_ports")
        assert response.status_code == 200
        ports = response.json()["ports"]
        assert isinstance(ports, list)
        for port in ports:
            assert set(port) == {"device", "description", "hwid"}


def test_init_payload_carries_the_deployed_array_layout():
    assert sim_server._deployed_program().name == json.loads(
        sim_server.default_contract_path().read_text(encoding="ascii"))["dsp_export"]["name"]
    with TestClient(sim_server.app) as client:
        assert client.post("/hw_connect", json={"transport": "emulator", "rows": 6, "columns": 6,
                                                "azimuth_min_deg": -40, "azimuth_max_deg": 40,
                                                "elevation_min_deg": -40, "elevation_max_deg": 40}).status_code == 200
        with client.websocket_connect("/realtime_hw") as websocket:
            initial = websocket.receive_json()
            assert initial["type"] == "init"
            layout = initial["array_layout"]
            assert layout["units"] == "mm"
            assert len(layout["positions"]) == 44
            assert layout["coordinate_system"]["x"] == "right"
            frame = websocket.receive_json()
            assert "array_layout" not in frame
        client.post("/hw_disconnect")


def test_bank_frames_carry_the_contract_beam_directions():
    bank = json.loads(sim_server.default_contract_path().read_text(encoding="ascii")).get("bank")
    if not bank:
        return
    with TestClient(sim_server.app) as client:
        assert client.post("/hw_connect", json={"transport": "emulator", "rows": 7, "columns": 7,
                                                "azimuth_min_deg": -40, "azimuth_max_deg": 40,
                                                "elevation_min_deg": -40, "elevation_max_deg": 40}).status_code == 200
        with client.websocket_connect("/realtime_hw") as websocket:
            payload = websocket.receive_json()
            for _ in range(20):
                if payload.get("bank"):
                    break
                payload = websocket.receive_json()
            assert payload["bank"]["azimuth_deg"] == [float(value) for value in bank["azimuth_deg"]]
            assert payload["bank"]["elevation_deg"] == [float(value) for value in bank["elevation_deg"]]
            assert "fit" in payload["bank"]
        client.post("/hw_disconnect")


def test_bank_frames_pass_the_firmware_fit_through():
    bank = json.loads(sim_server.default_contract_path().read_text(encoding="ascii")).get("bank")
    if not bank:
        return
    fit = {"azimuth_deg": 12.5, "elevation_deg": -3.0, "quality": 0.8, "sector": 24, "elapsed_us": 5000}
    frame = {"rows": bank["rows"], "columns": bank["columns"], "levels_raw": [1000] * len(bank["azimuth_deg"]),
             "maps": 3, "target": None, "fit": fit}
    result = sim_server._bank_with_directions(frame)
    assert result["fit"] == fit
    assert result["azimuth_deg"] == [float(value) for value in bank["azimuth_deg"]]
