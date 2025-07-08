#!/usr/bin/env python3
"""
Interactive ChArUco calibration capture for any number of Intel RealSense D415 cameras.

Key features
-------------
* **SPACE** captures one infrared frame from every connected camera.
* A set is accepted when the board appears in at least `--min_views` cameras.
* Live preview shows each feed in a compact grid:
  * red outlines around detected ArUco markers while the count is below `--threshold_charuco`,
  * green outlines **plus** a check‑mark when the camera meets the threshold.
* A second window displays an evolving **camera‑pair graph**:
  * nodes represent cameras,
  * an edge appears after at least one accepted image contains the board in both cameras,
  * edge labels indicate how many shared images exist for that pair.
* Only untouched infrared grayscale images are saved; overlays are for feedback only.
* Use `--hardware_reset` to recover unresponsive cameras.
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
RECONST_DIR = "reconstruction_images"
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
    p = argparse.ArgumentParser(description="Real‑time ChArUco capture and camera‑pair graph visualiser")
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


def serials_connected() -> list[str]:
    return [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]


def hard_reset():
    for d in rs.context().query_devices():
        d.hardware_reset()
    print("Hardware reset sent.  Re‑run after devices reconnect.")


def cfg_load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else CONFIG_DEFAULT.copy()


def cfg_save(cfg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))


def grid_stack(frames: list[np.ndarray]) -> np.ndarray:
    if len(frames) == 1:
        return frames[0]
    cols = ceil(sqrt(len(frames)))
    rows = ceil(len(frames) / cols)
    h, w = frames[0].shape[:2]
    blank = np.zeros_like(frames[0])
    rows_img = []
    for r in range(rows):
        row = []
        for c in range(cols):
            idx = r * cols + c
            row.append(frames[idx] if idx < len(frames) else blank)
        rows_img.append(np.hstack(row))
    return np.vstack(rows_img)


def draw_checkmark(img: np.ndarray):
    radius = int(min(img.shape[:2]) * 0.05)
    center = (radius + 10, radius + 10)
    cv2.circle(img, center, radius, (0, 255, 0), -1)
    thick = max(2, radius // 6)
    p1 = (center[0] - radius // 3, center[1])
    p2 = (center[0] - radius // 10, center[1] + radius // 3)
    p3 = (center[0] + radius // 2, center[1] - radius // 3)
    cv2.line(img, p1, p2, (255, 255, 255), thick)
    cv2.line(img, p2, p3, (255, 255, 255), thick)

# -----------------------------------------------------------------------------
# Graph visualisation helpers
# -----------------------------------------------------------------------------

def init_graph_window(labels):
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title("Camera‑pair graph")
    g = nx.Graph()
    g.add_nodes_from(labels)
    pos = nx.circular_layout(g)
    return fig, ax, g, pos


def update_graph(ax, g, pos, counts):
    ax.clear()
    # Add edges with count>0
    g.remove_edges_from(list(g.edges))
    for (a, b), c in counts.items():
        if c > 0:
            g.add_edge(a, b, weight=c)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#1f78b4", node_size=600)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=10, font_color="white")
    # draw edges
    if g.edges:
        widths = [2] * len(g.edges)
        nx.draw_networkx_edges(g, pos, ax=ax, width=widths)
        edge_labels = nx.get_edge_attributes(g, "weight")
        nx.draw_networkx_edge_labels(g, pos, ax=ax, edge_labels=edge_labels, font_size=9)
    ax.set_axis_off()
    ax.figure.canvas.draw()
    ax.figure.canvas.flush_events()

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()

    if args.hardware_reset:
        hard_reset()
        return

    serials = serials_connected()
    if not serials:
        print("No RealSense cameras connected")
        return

    # Graph data structures
    pair_counts = {(a, b): 0 for a, b in combinations(serials, 2)}
    fig, ax, graph, pos = init_graph_window(serials)

    out_root = Path(args.output_dir).resolve()
    cfg_path = Path(args.config_file).resolve()
    cfg = cfg_load(cfg_path)
    if args.num_imgs is not None:
        cfg["num_calibration_imgs"] = args.num_imgs

    # ArUco & ChArUco definitions
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    board = aruco.CharucoBoard_create(args.charuco_cols, args.charuco_rows, args.square_length, args.marker_length, aruco_dict)
    aruco_params = aruco.DetectorParameters_create()

    pipelines, profiles = [], []
    for s in serials:
        pipe = rs.pipeline()
        cfg_rs = rs.config()
        cfg_rs.enable_device(s)
        cfg_rs.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        cfg_rs.enable_stream(rs.stream.infrared, args.width, args.height, rs.format.y8, args.fps)
        prof = pipe.start(cfg_rs)
        pipelines.append(pipe)
        profiles.append(prof)

        sensor = prof.get_device().first_depth_sensor()
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, float(args.exposure))
        sensor.set_option(rs.option.gain, float(args.gain))

        cam_intr = cfg.setdefault("cams", {}).setdefault(s, {}).setdefault("intrinsics", {})
        ir_intr = rs.video_stream_profile(prof.get_stream(rs.stream.infrared)).get_intrinsics()
        col_intr = rs.video_stream_profile(prof.get_stream(rs.stream.color)).get_intrinsics()
        cam_intr.update({
            "img_size": [args.width, args.height],
            "focal_length": [col_intr.fx, col_intr.fy],
            "img_center": [col_intr.ppx, col_intr.ppy],
            "ir_focal_length": [ir_intr.fx, ir_intr.fy],
            "ir_img_center": [ir_intr.ppx, ir_intr.ppy],
        })

        (out_root / s / CALIB_DIR).mkdir(parents=True, exist_ok=True)

    cfg_save(cfg, cfg_path)
    print("SPACE to capture, ESC to quit.")

    saved = 0
    needed = cfg["num_calibration_imgs"]
    outline_thick = 4

    while saved < needed:
        frames = [p.wait_for_frames() for p in pipelines]
        annotated, raw_grays, good = [], [], []

        for fr in frames:
            ir = fr.first(rs.stream.infrared)
            gray = np.asanyarray(ir.get_data())
            raw_grays.append(gray)
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            cnt = 0 if ids is None else len(ids)
            ok = cnt >= args.threshold_charuco
            col = (0, 255, 0) if ok else (0, 0, 255)
            if cnt:
                for arr in corners:
                    pts = arr.reshape(-1, 2).astype(int)
                    cv2.polylines(vis, [pts], True, col, outline_thick)
            if ok:
                draw_checkmark(vis)
            annotated.append(vis)
            good.append(ok)

        cv2.imshow("Infrared", grid_stack(annotated))
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            if sum(good) < args.min_views:
                print(f"Board OK in {sum(good)} views, need {args.min_views}. Skipped.")
                continue
            # Increment pair counts for cameras that saw the board well
            good_cams = [serials[i] for i, flag in enumerate(good) if flag]
            for a, b in combinations(sorted(good_cams), 2):
                pair_counts[(a, b)] += 1
            update_graph(ax, graph, pos, pair_counts)

            # Save raw images
            saved += 1
            for s, raw in zip(serials, raw_grays):
                cv2.imwrite(str(out_root / s / CALIB_DIR / f"image_{saved}.jpg"), raw)
            print(f"Saved {saved}/{needed} (pair graph updated)")

    cv2.destroyAllWindows()
    for p in pipelines:
        p.stop()
    plt.ioff()
    plt.show()
    print("Capture session finished")


if __name__ == "__main__":
    main()
