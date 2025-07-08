#!/usr/bin/env python3
"""
Capture infrared frames that show a ChArUco board, store images for calibration, and write or update a
configuration JSON with camera intrinsics.

Press **SPACE** to grab one frame from every attached RealSense camera.  A set is accepted when the
board is detected in at least `--min_views` cameras (default 1).  The preview window arranges camera
feeds in a compact grid that automatically scales to any number of cameras: two‑by‑two for four
sensors, three‑by‑three for nine, and so on.

Images are written to
    <output_dir>/<serial>/calibration_images/image_<idx>.jpg

Use `--hardware_reset` if the cameras freeze.
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
# Utility helpers
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


def discover_serials():
    return [dev.get_info(rs.camera_info.serial_number) for dev in rs.context().query_devices()]


def hardware_reset():
    for dev in rs.context().query_devices():
        dev.hardware_reset()
    print("Hardware reset issued.  Re‑run the script once cameras reconnect.")


def load_cfg(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return CONFIG_DEFAULT.copy()


def save_cfg(cfg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))


def stack_grid(imgs: list[np.ndarray]) -> np.ndarray:
    """Return a compact grid image from a list of equally sized frames."""
    if len(imgs) == 1:
        return imgs[0]
    cols = ceil(sqrt(len(imgs)))
    rows = ceil(len(imgs) / cols)
    h, w = imgs[0].shape[:2]
    black = np.zeros_like(imgs[0])
    canvas_rows = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            idx = r * cols + c
            row_imgs.append(imgs[idx] if idx < len(imgs) else black)
        canvas_rows.append(np.hstack(row_imgs))
    return np.vstack(canvas_rows)


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main():
    args = make_parser().parse_args()

    if args.hardware_reset:
        hardware_reset()
        return

    serials = discover_serials()
    if not serials:
        print("No RealSense cameras detected.")
        return

    out_root = Path(args.output_dir).resolve()
    cfg_path = Path(args.config_file).resolve()
    cfg = load_cfg(cfg_path)
    if args.num_imgs is not None:
        cfg["num_calibration_imgs"] = args.num_imgs

    # Prepare ChArUco board
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    board = aruco.CharucoBoard_create(
        squaresX=args.charuco_cols,
        squaresY=args.charuco_rows,
        squareLength=args.square_length,
        markerLength=args.marker_length,
        dictionary=aruco_dict,
    )
    aruco_params = aruco.DetectorParameters_create()

    # Start pipelines
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

        # Write intrinsics if absent
        cam_cfg = cfg.setdefault("cams", {}).setdefault(s, {}).setdefault("intrinsics", {})
        ir_intr = rs.video_stream_profile(prof.get_stream(rs.stream.infrared)).get_intrinsics()
        color_intr = rs.video_stream_profile(prof.get_stream(rs.stream.color)).get_intrinsics()
        cam_cfg.update({
            "img_size": [args.width, args.height],
            "focal_length": [color_intr.fx, color_intr.fy],
            "img_center": [color_intr.ppx, color_intr.ppy],
            "ir_focal_length": [ir_intr.fx, ir_intr.fy],
            "ir_img_center": [ir_intr.ppx, ir_intr.ppy],
        })

        (out_root / s / CALIB_DIR).mkdir(parents=True, exist_ok=True)
        (out_root / s / RECONST_DIR).mkdir(parents=True, exist_ok=True)

    save_cfg(cfg, cfg_path)
    print("Connected cameras:", ", ".join(serials))
    print("Press SPACE to capture, ESC to quit.")

    captured = 0
    while captured < cfg["num_calibration_imgs"]:
        frames = [p.wait_for_frames() for p in pipelines]
        imgs_ir, board_ok = [], []
        for fr in frames:
            ir = fr.first(rs.stream.infrared)
            img = np.asanyarray(ir.get_data())
            imgs_ir.append(img)
            corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
            board_ok.append(ids is not None and len(ids) >= args.threshold_charuco)

        cv2.imshow("Infrared", stack_grid(imgs_ir))
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            if sum(board_ok) < args.min_views:
                print(f"Board visible in {sum(board_ok)} views < {args.min_views}. Frame skipped.")
                continue
            captured += 1
            for s, img in zip(serials, imgs_ir):
                fname = out_root / s / CALIB_DIR / f"image_{captured}.jpg"
                cv2.imwrite(str(fname), img)
            print(f"Saved frame {captured}/{cfg['num_calibration_imgs']} (board in {sum(board_ok)} cameras)")

    cv2.destroyAllWindows()
    for p in pipelines:
        p.stop()
    print("Finished.")


if __name__ == "__main__":
    main()
