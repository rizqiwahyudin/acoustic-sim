"""
beam_scan_pyqtgraph.py

Receives beam scan data from MAX78000 over serial (UART).
Plots a live-updating polar-style plot using pyqtgraph.

Protocol from MAX78000:
    P,<angle_deg>,<level_raw>   raw 8.24 level value
    SCAN_DONE                   end of one full 360 scan
"""

import argparse
import sys
import time
import threading
import serial
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore


def parse_args():
    parser = argparse.ArgumentParser(description="Beam scan polar plot (pyqtgraph)")
    parser.add_argument("--port", type=str, required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--num-angles", type=int, default=72)
    parser.add_argument("--angle-step", type=int, default=5)
    parser.add_argument("--db", action="store_true")
    parser.add_argument("--db-floor", type=float, default=-40.0)
    parser.add_argument("--avg", type=int, default=4, help="Number of scans to average")
    return parser.parse_args()


class SerialReader(threading.Thread):
    def __init__(self, port, baud, num_angles, angle_step):
        super().__init__(daemon=True)
        self.ser = serial.Serial(port, baud, timeout=0)
        self.num_angles = num_angles
        self.angle_step = angle_step
        self.levels = np.zeros(num_angles, dtype=np.float64)
        self.latest_scan = np.zeros(num_angles, dtype=np.float64)
        self.scan_ready = False
        self.scan_count = 0
        self.running = True
        self.lock = threading.Lock()
        self.rx_buf = b""

    def run(self):
        while self.running:
            n = self.ser.in_waiting
            if n > 0:
                data = self.ser.read(n)
                self.rx_buf += data
                while b"\n" in self.rx_buf:
                    line_bytes, self.rx_buf = self.rx_buf.split(b"\n", 1)
                    line = line_bytes.decode(errors="ignore").strip()
                    self._process_line(line)
            else:
                time.sleep(0.001)

    def _process_line(self, line):
        if line.startswith("P,"):
            parts = line.split(",")
            if len(parts) == 3:
                try:
                    angle = int(parts[1])
                    raw = int(parts[2])
                    idx = angle // self.angle_step
                    if 0 <= idx < self.num_angles:
                        self.levels[idx] = raw / 16777216.0
                except ValueError:
                    pass

        elif line == "SCAN_DONE":
            with self.lock:
                self.latest_scan[:] = self.levels
                self.scan_count += 1
                self.scan_ready = True

        elif line.startswith("T,"):
            print(f"FW: {line}")

    def get_scan(self):
        with self.lock:
            if not self.scan_ready:
                return None, self.scan_count
            self.scan_ready = False
            return self.latest_scan.copy(), self.scan_count

    def stop(self):
        self.running = False
        self.ser.close()


def main():
    args = parse_args()

    num_angles = args.num_angles
    angle_step = args.angle_step
    angles_deg = np.arange(0, num_angles * angle_step, angle_step, dtype=float)
    angles_rad = np.deg2rad(angles_deg)

    cos_a = np.cos(angles_rad)
    sin_a = np.sin(angles_rad)

    interp_factor = 4
    num_interp = num_angles * interp_factor
    angles_interp_rad = np.linspace(0, 2 * np.pi, num_interp, endpoint=False)
    cos_interp = np.cos(angles_interp_rad)
    sin_interp = np.sin(angles_interp_rad)

    avg_count = max(1, args.avg)
    scan_history = np.zeros((avg_count, num_angles), dtype=np.float64)
    history_idx = [0]
    history_filled = [0]

    source = SerialReader(args.port, args.baud, num_angles, angle_step)
    source.start()

    app = QtWidgets.QApplication(sys.argv)

    win = pg.GraphicsLayoutWidget(title="Beam Scan")
    win.resize(900, 900)
    win.setBackground("k")

    plot = win.addPlot()
    plot.setAspectLocked(True)
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.hideAxis("bottom")
    plot.hideAxis("left")

    for r in [0.25, 0.5, 0.75, 1.0]:
        t = np.linspace(0, 2 * np.pi, 361)
        plot.plot(r * np.cos(t), r * np.sin(t), pen=pg.mkPen((60, 60, 60), width=1))

    for a in range(0, 360, 30):
        ar = np.radians(a)
        plot.plot([0, 1.1 * np.cos(ar)], [0, 1.1 * np.sin(ar)],
                  pen=pg.mkPen((40, 40, 40), width=1))

    for a in range(0, 360, 30):
        ar = np.radians(a)
        label = pg.TextItem(f"{a}", color=(120, 120, 120), anchor=(0.5, 0.5))
        label.setPos(1.18 * np.cos(ar), 1.18 * np.sin(ar))
        plot.addItem(label)

    curve = plot.plot([], [], pen=pg.mkPen("c", width=2))

    peak_dot = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None),
                                  brush=pg.mkBrush(255, 50, 50))
    plot.addItem(peak_dot)

    peak_line = plot.plot([], [],
                          pen=pg.mkPen("r", width=2,
                                       style=QtCore.Qt.PenStyle.DashLine))

    info = pg.TextItem("Waiting for data...", color="w", anchor=(0, 0))
    info.setPos(-1.3, 1.3)
    plot.addItem(info)

    perf_text = pg.TextItem("", color=(100, 100, 100), anchor=(0, 0))
    perf_text.setPos(-1.3, -1.25)
    plot.addItem(perf_text)

    plot.setXRange(-1.4, 1.4)
    plot.setYRange(-1.4, 1.4)

    last_perf_time = time.perf_counter()
    frames_since_perf = [0]
    scans_since_perf = [0]

    def update():
        nonlocal last_perf_time

        scan, scan_count = source.get_scan()
        if scan is None:
            return

        scan_history[history_idx[0] % avg_count] = scan
        history_idx[0] += 1
        if history_filled[0] < avg_count:
            history_filled[0] += 1

        levels = np.mean(scan_history[:history_filled[0]], axis=0)
        levels = np.maximum(levels, 1e-12)

        scans_since_perf[0] += 1
        frames_since_perf[0] += 1

        if args.db:
            vals_db = 20.0 * np.log10(levels)
            vals_db = np.clip(vals_db, args.db_floor, 0.0)
            vals = (vals_db - args.db_floor) / abs(args.db_floor)
        else:
            mx = np.max(levels)
            vals = levels / mx if mx > 0 else levels

        vals_wrapped = np.append(vals, vals[0])
        angles_wrapped = np.append(angles_rad, 2 * np.pi)
        vals_interp = np.interp(angles_interp_rad, angles_wrapped, vals_wrapped)

        x = vals_interp * cos_interp
        y = vals_interp * sin_interp

        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])
        curve.setData(x_closed, y_closed)

        peak_idx = int(np.argmax(levels))
        peak_angle = angles_deg[peak_idx]
        peak_level = levels[peak_idx]
        peak_val = vals[peak_idx]

        px = peak_val * cos_a[peak_idx]
        py = peak_val * sin_a[peak_idx]
        peak_dot.setData([px], [py])
        peak_line.setData([0, px * 1.3], [0, py * 1.3])

        info.setText(
            f"Scan #{scan_count}  |  Peak: {peak_angle:.0f} deg  |  "
            f"Level: {peak_level:.6f}  |  Avg: {history_filled[0]}"
        )

        now = time.perf_counter()
        if now - last_perf_time >= 1.0:
            fps = frames_since_perf[0] / (now - last_perf_time)
            sps = scans_since_perf[0] / (now - last_perf_time)
            perf_text.setText(
                f"GUI: {fps:.1f} FPS  |  Scans: {sps:.1f}/sec  |  Avg window: {history_filled[0]}"
            )
            frames_since_perf[0] = 0
            scans_since_perf[0] = 0
            last_perf_time = now

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(10)

    win.show()
    print("Running. Close window to exit.")

    try:
        sys.exit(app.exec_())
    finally:
        source.stop()


if __name__ == "__main__":
    main()