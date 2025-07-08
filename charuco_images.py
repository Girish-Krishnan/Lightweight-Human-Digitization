#!/usr/bin/env python3
"""
Interactive ChArUco calibration capture for an arbitrary array of Intel RealSense D415 cameras.

Live feedback
-------------
* **SPACE** captures a synchronized infrared frame from every camera.
* A capture is accepted when the board is present in at least `--min_views` cameras.
* The preview grid shows, per camera:
  * red outlines while the marker count is below `--threshold_charuco`,
  * green outlines plus a tick after the count reaches the threshold,
  * a colour‑coded banner with the camera serial top‑right (never overlaps the tick).
* A second window hosts a colour‑coded camera graph:
  * each node has the same colour as its preview banner,
  * an edge appears once the pair has at least one shared capture,
  * edge label increments with every additional shared image.
* Only raw infrared grayscale frames are saved.
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
import matplotlib as mpl
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
    p.add_argument("--num_imgs", type=int, default=30)
    p.add_argument("--hardware_reset", action="store_true")
    return p


def connected_serials():
    return [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]


def hardware_reset():
    for d in rs.context().query_devices():
        d.hardware_reset()
    print("Hardware reset issued.  Re‑run after cameras reconnect.")


def load_cfg(path: Path):
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
    out_rows = []
    for r in range(rows):
        row = []
        for c in range(cols):
            idx = r * cols + c
            row.append(frames[idx] if idx < len(frames) else blk)
        out_rows.append(np.hstack(row))
    return np.vstack(out_rows)


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


def banner(img, text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    w = img.shape[1]
    x0 = w - (tw + 15)
    cv2.rectangle(img, (x0, 0), (w, th + 12), color, -1)
    cv2.putText(img, text, (x0 + 5, th + 6), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

# -------------------- graph helpers --------------------

def init_graph(serials, color_map):
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title("Camera graph")
    g = nx.Graph()
    g.add_nodes_from(serials)
    pos = nx.circular_layout(g)
    nx.draw_networkx_nodes(g, pos, node_color=[color_map[s] for s in serials], node_size=4500, node_shape='s', ax=ax)
    nx.draw_networkx_labels(g, pos, labels={s: s for s in serials}, font_size=7, font_color="white", ax=ax)
    ax.set_axis_off()
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, g, pos


def update_graph(ax, g, pos, counts, color_map):
    ax.clear()
    g.remove_edges_from(list(g.edges))
    for (a, b), n in counts.items():
        if n > 0:
            g.add_edge(a, b, weight=n)
    nx.draw_networkx_nodes(g, pos, node_color=[color_map[s] for s in g.nodes], node_size=4500, node_shape='s', ax=ax)
    nx.draw_networkx_labels(g, pos, labels={s: s for s in g.nodes}, font_size=7, font_color="white", ax=ax)
    if g.edges:
        nx.draw_networkx_edges(g, pos, width=2, ax=ax)
        nx.draw_networkx_edge_labels(g, pos, edge_labels=nx.get_edge_attributes(g, "weight"), font_size=8, ax=ax)
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

    # assign distinct colours using tab10 / tab20 cycle
    cmap = mpl.cm.get_cmap('tab20')
    color_map = {s: tuple(int(255 * c) for c in cmap(i % 20)[:3]) for i, s in enumerate(serials)}
    node_colors_plot = {s: mpl.colors.rgb2hex(np.array(color_map[s]) / 255) for s in serials}

    pair_counts = {tuple(sorted((a, b))): 0 for a, b in combinations(serials, 2)}
    fig, ax, graph, pos = init_graph(serials, node_colors_plot)

    root = Path(args.output_dir).resolve()
    cfg_path = Path(args.config_file).resolve()
    cfg = load_cfg(cfg_path)
    if args.num_imgs is not None:
        cfg["num_calibration_imgs"] = args.num_imgs

    dictionary = aruco.Dictionary_get(aruco.DICT_5X5_250)
    detector_params = aruco.DetectorParameters_create()

    pipes, sensors = [], []
    for s in serials:
        pipe = rs.pipeline()
        rcfg = rs.config()
        rcfg.enable_device(s)
        rcfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        rcfg.enable_stream(rs.stream.infrared, args.width, args.height, rs.format.y8, args.fps)
        pipe.start(rcfg)

        prof = pipe.get_active_profile()
        color_profile = rs.video_stream_profile(prof.get_stream(rs.stream.color))
        ir_profile = rs.video_stream_profile(prof.get_stream(rs.stream.infrared))
        color_intr = color_profile.get_intrinsics()
        ir_intr = ir_profile.get_intrinsics()

        # store intrinsics in configuration file
        intr_block = cfg.setdefault("cams", {}).setdefault(s, {}).setdefault("intrinsics", {})
        intr_block.update({
            "img_size": [args.width, args.height],
            "focal_length": [color_intr.fx, color_intr.fy],
            "img_center": [color_intr.ppx, color_intr.ppy],
            "ir_focal_length": [ir_intr.fx, ir_intr.fy],
            "ir_img_center": [ir_intr.ppx, ir_intr.ppy],
        })

        sen = prof.get_device().first_depth_sensor()
        sen.set_option(rs.option.enable_auto_exposure, 0)
        sen.set_option(rs.option.exposure, float(args.exposure))
        sen.set_option(rs.option.gain, float(args.gain))

        pipes.append(pipe)
        sensors.append(sen)

        (root / s / CALIB_DIR).mkdir(parents=True, exist_ok=True)
        (root / s / "reconstruction_images").mkdir(parents=True, exist_ok=True)

    save_cfg(cfg, cfg_path)
    needed, saved = cfg["num_calibration_imgs"], 0
    outline = 4

    while saved < needed:
        frames = [p.wait_for_frames() for p in pipes]
        previews, raws, ok = [], [], []
        for ser, fr in zip(serials, frames):
            gray = np.asanyarray(fr.first(rs.stream.infrared).get_data())
            raws.append(gray)
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            banner(vis, ser, color_map[ser])
            corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=detector_params)
            cnt = 0 if ids is None else len(ids)
            good = cnt >= args.threshold_charuco
            col = (0, 255, 0) if good else (0, 0, 255)
            if cnt:
                for arr in corners:
                    cv2.polylines(vis, [arr.reshape(-1, 2).astype(int)], True, col, outline)
            if good:
                draw_check(vis)
            previews.append(vis)
            ok.append(good)

        cv2.imshow("Infrared", stack_grid(previews))
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            good_serials = [s for s, g in zip(serials, ok) if g]
            if len(good_serials) < args.min_views:
                print(f"Need {args.min_views} good views, got {len(good_serials)}. Skipped.")
                continue
            for a, b in combinations(sorted(good_serials), 2):
                pair_counts[tuple(sorted((a, b)))] += 1
            update_graph(ax, graph, pos, pair_counts, node_colors_plot)
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
