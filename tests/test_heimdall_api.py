from __future__ import annotations

from fastapi.testclient import TestClient

import sim_server


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

        response = client.post("/hw_disconnect")
        assert response.status_code == 200
        assert not client.get("/hw_status").json()["available"]