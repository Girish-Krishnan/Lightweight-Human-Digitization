#!/usr/bin/env python3
"""
Capture infrared frames that show a ChArUco board, store images for calibration, and write or update a
configuration JSON with camera intrinsics.

*  **SPACE** – capture one frame from every attached RealSense camera.
*  A capture is accepted when the board is detected in at least `--min_views` cameras (default 1).
*  Each live feed is displayed in a compact grid (2 × 2 for four cameras, 3 × 3 for nine, …).
*  Detected ArUco markers are drawn **red** while the count is below `--threshold_charuco`, and turn
   **green** when the count meets or exceeds the threshold—helping you place the board correctly.
*  Images are saved as `<output_dir>/<serial>/calibration_images/image_<idx>.jpg`.
*  Use `--hardware_reset` to recover frozen cameras.
"""
import json
import argparse
from math import ceil, sqrt
from pathlib import Path
from datetime import datetime

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

def make_parser():
    p = argparse.ArgumentParser(description="Capture ChArUco calibration frames from multiple RealSense cameras")
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


def serials_rs() -> list[str]:
    return [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]


def reset_hardware():
    for d in rs.context().query_devices():
        d.hardware_reset()
    print("Hardware reset sent.  Wait for cameras to reconnect and rerun the script.")


def load_cfg(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return CONFIG_DEFAULT.copy()


def save_cfg(cfg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))


def stack_grid(frames: list[np.ndarray]) -> np.ndarray:
    if len(frames) == 1:
        return frames[0]
    cols = ceil(sqrt(len(frames)))
    rows = ceil(len(frames) / cols)
    h, w = frames[0].shape[:2]
    blank = np.zeros_like(frames[0])
    rows_img = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            idx = r * cols + c
            row_cells.append(frames[idx] if idx < len(frames) else blank)
        rows_img.append(np.hstack(row_cells))
    return np.vstack(rows_img)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = make_parser().parse_args()

    if args.hardware_reset:
        reset_hardware()
        return

    serials = serials_rs()
    if not serials:
        print("No RealSense cameras found.")
        return

    out_root = Path(args.output_dir).resolve()
    cfg_path = Path(args.config_file).resolve()
    cfg = load_cfg(cfg_path)
    if args.num_imgs is not None:
        cfg["num_calibration_imgs"] = args.num_imgs

    # ChArUco definitions
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    board = aruco.CharucoBoard_create(
        squaresX=args.charuco_cols,
        squaresY=args.charuco_rows,
        squareLength=args.square_length,
        markerLength=args.marker_length,
        dictionary=aruco_dict,
    )
    aruco_params = aruco.DetectorParameters_create()

    # Start each camera
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

        intr_block = cfg.setdefault("cams", {}).setdefault(s, {}).setdefault("intrinsics", {})
        ir_intr = rs.video_stream_profile(prof.get_stream(rs.stream.infrared)).get_intrinsics()
        color_intr = rs.video_stream_profile(prof.get_stream(rs.stream.color)).get_intrinsics()
        intr_block.update(
            {
                "img_size": [args.width, args.height],
                "focal_length": [color_intr.fx, color_intr.fy],
                "img_center": [color_intr.ppx, color_intr.ppy],
                "ir_focal_length": [ir_intr.fx, ir_intr.fy],
                "ir_img_center": [ir_intr.ppx, ir_intr.ppy],
            }
        )

        (out_root / s / CALIB_DIR).mkdir(parents=True, exist_ok=True)
        (out_root / s / RECONST_DIR).mkdir(parents=True, exist_ok=True)

    save_cfg(cfg, cfg_path)
    print("Connected cameras:", ", ".join(serials))
    print("Press SPACE to capture, ESC to quit.")

    captured = 0
    total_needed = cfg["num_calibration_imgs"]

    while captured < total_needed:
        frames = [p.wait_for_frames() for p in pipelines]
        annotated, good_views = [], []

        for fr in frames:
            ir_frame = fr.first(rs.stream.infrared)
            gray = np.asanyarray(ir_frame.get_data())
            img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            count = 0 if ids is None else len(ids)
            meets = count >= args.threshold_charuco
            color = (0, 255, 0) if meets else (0, 0, 255)  # green or red
            if count:
                aruco.drawDetectedMarkers(img_color, corners, borderColor=color)
            annotated.append(img_color)
            good_views.append(meets)

        cv2.imshow("Infrared", stack_grid(annotated))
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            if sum(good_views) < args.min_views:
                print(f"Only {sum(good_views)} cameras meet the threshold (< {args.min_views}). Frame skipped.")
                continue
            captured += 1
            for s, img in zip(serials, annotated):
                cv2.imwrite(str(out_root / s / CALIB_DIR / f"image_{captured}.jpg"), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            print(f"Saved frame {captured}/{total_needed} (good in {sum(good_views)} cameras)")

    cv2.destroyAllWindows()
    for p in pipelines:
        p.stop()
    print("Done.")


if __name__ == "__main__":
    main()
