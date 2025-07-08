#!/usr/bin/env python3
"""
Interactive ChArUco calibration capture for any number of Intel RealSense D415 cameras.

* **SPACE** captures a synchronized infrared frame from every camera.
* A capture is accepted when the ChArUco board appears in at least `--min_views` cameras.
* Preview grid per camera:
  * red marker outlines when count < `--threshold_charuco`,
  * green outlines plus a tick when the camera passes the threshold,
  * serial number shown top‑left in a red banner for quick identification.
* Realtime **camera‑pair graph** in a second window:
  * rectangular nodes show full serial numbers,
  * undirected edge appears once a pair has ≥ 1 shared capture,
  * edge label counts shared captures and increments live.
* Only raw infrared grayscale images are saved; overlays are for feedback only.
* Use `--hardware_reset` if cameras freeze.
"""
import json
import argparse
from math import ceil, sqrt
from itertools import combinations
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import networkx as nx

CALIB_DIR = "calibration_images"
CONFIG_DEFAULT = {
    "checkerboard_size_mm": 60,
    "checkerboard_dimensions": [8, 11],
    "num_calibration_imgs": 30,
    "img_file_type": ".jpg",
    "threshold": 5,
    "cams": {}
}

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="ChArUco capture with live camera‑pair graph")
    p.add_argument("--output_dir", default="./Capture_Data")
    p.add_argument("--config_file", default="./configuration_parameters.json")
    p.add_argument("--charuco_rows", type=int, default=9)
    p.add_argument("--charuco_cols", type=int, default=12)
    p.add_argument("--square_length", type=float, default=0.060)
    p.add_argument("--marker_length", type=float, default=0.044)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--exposure", type=int, default=70000)
    p.add_argument("--gain", type=int, default=30)
    p.add_argument("--threshold_charuco", type=int, default=20)
    p.add_argument("--min_views", type=int, default=1)
    p.add_argument("--num_imgs", type=int)
    p.add_argument("--hardware_reset", action="store_true")
    return p


def connected_serials():
    return [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]


def hardware_reset():
    for d in rs.context().query_devices():
        d.hardware_reset()
    print("Hardware reset issued.  Re‑run after cameras reconnect.")


def load_cfg(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else CONFIG_DEFAULT.copy()


def save_cfg(cfg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))


def stack_grid(frames):
    if len(frames) == 1:
        return frames[0]
    cols = ceil(sqrt(len(frames)))
    rows = ceil(len(frames) / cols)
    h, w = frames[0].shape[:2]
    blk = np.zeros_like(frames[0])
    rows_out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            idx = r * cols + c
            row.append(frames[idx] if idx < len(frames) else blk)
        rows_out.append(np.hstack(row))
    return np.vstack(rows_out)


def draw_check(img):
    rad = int(min(img.shape[:2]) * 0.05)
    ctr = (rad + 10, rad + 10)
    cv2.circle(img, ctr, rad, (0, 255, 0), -1)
    t = max(2, rad // 6)
    p1 = (ctr[0] - rad // 3, ctr[1])
    p2 = (ctr[0] - rad // 10, ctr[1] + rad // 3)
    p3 = (ctr[0] + rad // 2, ctr[1] - rad // 3)
    cv2.line(img, p1, p2, (255, 255, 255), t)
    cv2.line(img, p2, p3, (255, 255, 255), t)


def banner_text(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (0, 0), (tw + 10, th + 10), (0, 0, 255), -1)
    cv2.putText(img, text, (5, th + 5), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

# --------------------- graph helpers ---------------------

def init_graph(serials):
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title("Camera graph")
    g = nx.Graph()
    g.add_nodes_from(serials)
    pos = nx.circular_layout(g)
    return fig, ax, g, pos


def update_graph(ax, g, pos, counts):
    ax.clear()
    g.remove_edges_from(list(g.edges))
    for (a, b), v in counts.items():
        if v > 0:
            g.add_edge(a, b, weight=v)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#1f78b4", node_size=4500, node_shape='s')
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=7, font_color="white")
    if g.edges:
        nx.draw_networkx_edges(g, pos, ax=ax, width=2)
        nx.draw_networkx_edge_labels(g, pos, ax=ax, edge_labels=nx.get_edge_attributes(g, "weight"), font_size=8)
    ax.set_axis_off()
    ax.figure.canvas.draw()
    ax.figure.canvas.flush_events()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()

    if args.hardware_reset:
        hardware_reset()
        return

    serials = connected_serials()
    if not serials:
        print("No cameras detected")
        return

    pair_counts = {tuple(sorted((a, b))): 0 for a, b in combinations(serials, 2)}
    fig, ax, graph, pos = init_graph(serials)

    root = Path(args.output_dir).resolve()
    cfg_path = Path(args.config_file).resolve()
    cfg = load_cfg(cfg_path)
    if args.num_imgs is not None:
        cfg["num_calibration_imgs"] = args.num_imgs

    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    params = aruco.DetectorParameters_create()

    pipes, profiles = [], []
    for s in serials:
        pipe = rs.pipeline()
        crs = rs.config()
        crs.enable_device(s)
        crs.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        crs.enable_stream(rs.stream.infrared, args.width, args.height, rs.format.y8, args.fps)
        prof = pipe.start(crs)
        pipes.append(pipe)
        profiles.append(prof)
        dev = prof.get_device().first_depth_sensor()
        dev.set_option(rs.option.enable_auto_exposure, 0)
        dev.set_option(rs.option.exposure, float(args.exposure))
        dev.set_option(rs.option.gain, float(args.gain))
        (root / s / CALIB_DIR).mkdir(parents=True, exist_ok=True)

    save_cfg(cfg, cfg_path)
    needed = cfg["num_calibration_imgs"]
    saved = 0
    thick = 4

    while saved < needed:
        frames = [p.wait_for_frames() for p in pipes]
        previews, raws, ok_flags = [], [], []
        for idx, fr in enumerate(frames):
            ser = serials[idx]
            gray = np.asanyarray(fr.first(rs.stream.infrared).get_data())
            raws.append(gray)
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            banner_text(vis, ser)
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)
            cnt = 0 if ids is None else len(ids)
            good = cnt >= args.threshold_charuco
            col = (0, 255, 0) if good else (0, 0, 255)
            if cnt:
                for arr in corners:
                    cv2.polylines(vis, [arr.reshape(-1, 2).astype(int)], True, col, thick)
            if good:
                draw_check(vis)
            previews.append(vis)
            ok_flags.append(good)

        cv2.imshow("Infrared", stack_grid(previews))
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            good_serials = [s for s, ok in zip(serials, ok_flags) if ok]
            if len(good_serials) < args.min_views:
                print(f"Need {args.min_views} good views, got {len(good_serials)}. Skipped.")
                continue
            for a, b in combinations(sorted(good_serials), 2):
                pair_counts[tuple(sorted((a, b)))] += 1
            update_graph(ax, graph, pos, pair_counts)
            saved += 1
            for s, img in zip(serials, raws):
                cv2.imwrite(str(root / s / CALIB_DIR / f"image_{saved}.jpg"), img)
            print(f"Saved {saved}/{needed}")

    cv2.destroyAllWindows()
    for p in pipes:
        p.stop()
    plt.ioff()
    plt.show()
    print("Session finished")


if __name__ == "__main__":
    main()
