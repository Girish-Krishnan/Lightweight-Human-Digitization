#!/usr/bin/env python3
"""
Interactive ChArUco calibration capture for any number of Intel RealSense D415 cameras.

* Press **SPACE** to record one frame from every connected camera.
* A set is accepted when the board is detected in at least `--min_views` cameras.
* Each live feed shows
  * red outlines while the marker count is below `--threshold_charuco`,
  * green outlines **plus** a green circle‑and‑check mark once the count meets the threshold.
* Feeds are arranged in a square grid that scales automatically.
* **Only the untouched infrared grayscale images are written to disk**—visual overlays exist only in the preview.
* Use `--hardware_reset` if a camera locks up.
"""
import json
import argparse
from math import ceil, sqrt
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs

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
# Helper functions
# -----------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Capture ChArUco calibration frames with visual feedback")
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


def connected_serials() -> list[str]:
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


def grid_stack(frames: list[np.ndarray]) -> np.ndarray:
    if len(frames) == 1:
        return frames[0]
    cols = ceil(sqrt(len(frames)))
    rows = ceil(len(frames) / cols)
    h, w = frames[0].shape[:2]
    black = np.zeros_like(frames[0])
    stacked_rows = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            idx = r * cols + c
            row_cells.append(frames[idx] if idx < len(frames) else black)
        stacked_rows.append(np.hstack(row_cells))
    return np.vstack(stacked_rows)


def draw_checkmark(img: np.ndarray):
    radius = int(min(img.shape[:2]) * 0.05)
    center = (radius + 10, radius + 10)
    cv2.circle(img, center, radius, (0, 255, 0), -1)
    thickness = max(2, radius // 6)
    p1 = (center[0] - radius // 3, center[1])
    p2 = (center[0] - radius // 10, center[1] + radius // 3)
    p3 = (center[0] + radius // 2, center[1] - radius // 3)
    cv2.line(img, p1, p2, (255, 255, 255), thickness)
    cv2.line(img, p2, p3, (255, 255, 255), thickness)

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()

    if args.hardware_reset:
        hardware_reset()
        return

    serials = connected_serials()
    if not serials:
        print("No RealSense cameras found.")
        return

    out_root = Path(args.output_dir).resolve()
    cfg_path = Path(args.config_file).resolve()
    cfg = load_cfg(cfg_path)
    if args.num_imgs is not None:
        cfg["num_calibration_imgs"] = args.num_imgs

    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    board = aruco.CharucoBoard_create(
        squaresX=args.charuco_cols,
        squaresY=args.charuco_rows,
        squareLength=args.square_length,
        markerLength=args.marker_length,
        dictionary=aruco_dict,
    )
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

        intrinsic_block = cfg.setdefault("cams", {}).setdefault(s, {}).setdefault("intrinsics", {})
        ir_intr = rs.video_stream_profile(prof.get_stream(rs.stream.infrared)).get_intrinsics()
        col_intr = rs.video_stream_profile(prof.get_stream(rs.stream.color)).get_intrinsics()
        intrinsic_block.update({
            "img_size": [args.width, args.height],
            "focal_length": [col_intr.fx, col_intr.fy],
            "img_center": [col_intr.ppx, col_intr.ppy],
            "ir_focal_length": [ir_intr.fx, ir_intr.fy],
            "ir_img_center": [ir_intr.ppx, ir_intr.ppy],
        })

        (out_root / s / CALIB_DIR).mkdir(parents=True, exist_ok=True)
        (out_root / s / RECONST_DIR).mkdir(parents=True, exist_ok=True)

    save_cfg(cfg, cfg_path)
    print("Connected cameras:", ", ".join(serials))
    print("SPACE to capture, ESC to quit.")

    saved = 0
    needed = cfg["num_calibration_imgs"]
    outline_thickness = 4

    while saved < needed:
        frames = [p.wait_for_frames() for p in pipelines]
        annotated_views, raw_grays, ok_flags = [], [], []

        for fr in frames:
            ir = fr.first(rs.stream.infrared)
            gray = np.asanyarray(ir.get_data())
            raw_grays.append(gray)  # store untouched frame for saving later

            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            count = 0 if ids is None else len(ids)
            good = count >= args.threshold_charuco
            color = (0, 255, 0) if good else (0, 0, 255)

            if count:
                for arr in corners:
                    pts = arr.reshape(-1, 2).astype(int)
                    cv2.polylines(vis, [pts], True, color, outline_thickness)
            if good:
                draw_checkmark(vis)
            annotated_views.append(vis)
            ok_flags.append(good)

        cv2.imshow("Infrared", grid_stack(annotated_views))
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            if sum(ok_flags) < args.min_views:
                print(f"Board acceptable in {sum(ok_flags)} views, need {args.min_views}. Skipped.")
                continue
            saved += 1
            for s, raw in zip(serials, raw_grays):
                cv2.imwrite(str(out_root / s / CALIB_DIR / f"image_{saved}.jpg"), raw)
            print(f"Saved {saved}/{needed}   (good in {sum(ok_flags)} cameras)")

    cv2.destroyAllWindows()
    for p in pipelines:
        p.stop()
    print("Calibration capture complete")


if __name__ == "__main__":
    main()
