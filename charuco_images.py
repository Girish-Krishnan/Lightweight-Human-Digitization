#!/usr/bin/env python3
"""
Capture infrared frames that show a ChArUco board, store images for calibration, and write or update a
configuration JSON with camera intrinsics.

Press **SPACE** to grab one frame from **every** attached RealSense camera.  The script now accepts a
frame if the board is detected in *at least* `--min_views` cameras (default 1), so you are not required
to keep the board visible to every sensor at the same time.

Folders created per camera:
    <output_dir>/<serial>/calibration_images/image_<idx>.jpg

Use `--hardware_reset` when cameras become unresponsive.
"""
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs

CALIB_DIR_NAME = "calibration_images"
RECONST_DIR_NAME = "reconstruction_images"
CONFIG_DEFAULT = {
    "checkerboard_size_mm": 60,
    "checkerboard_dimensions": [8, 11],
    "num_calibration_imgs": 30,
    "img_file_type": ".jpg",
    "threshold": 5,
    "cams": {}
}

def parse_args():
    p = argparse.ArgumentParser(description="Grab ChArUco calibration frames from Intel RealSense cameras")
    p.add_argument("--output_dir", default="./Capture_Data", help="Root folder for captured data")
    p.add_argument("--config_file", default="./configuration_parameters.json", help="Path for the JSON config")
    p.add_argument("--charuco_rows", type=int, default=9)
    p.add_argument("--charuco_cols", type=int, default=12)
    p.add_argument("--square_length", type=float, default=0.060)
    p.add_argument("--marker_length", type=float, default=0.044)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--exposure", type=int, default=70000)
    p.add_argument("--gain", type=int, default=30)
    p.add_argument("--threshold_charuco", type=int, default=20, help="minimum visible Charuco ids per camera")
    p.add_argument("--min_views", type=int, default=1, help="accept a frame if the board is seen by THIS many cameras")
    p.add_argument("--num_imgs", type=int, help="override num_calibration_imgs in the config")
    p.add_argument("--hardware_reset", action="store_true", help="reset all connected cameras and quit")
    return p.parse_args()

def discover_serials():
    ctx = rs.context()
    return [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]

def reset_hardware():
    ctx = rs.context()
    for dev in ctx.query_devices():
        dev.hardware_reset()
    print("Reset complete.  Please unplug and replug cameras if they do not reappear.")

def load_or_create_config(path: Path) -> dict:
    if path.exists():
        with path.open() as f:
            cfg = json.load(f)
    else:
        cfg = CONFIG_DEFAULT.copy()
    for k, v in CONFIG_DEFAULT.items():
        cfg.setdefault(k, v)
    return cfg

def save_config(cfg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(cfg, f, indent=2)


def main():
    args = parse_args()

    if args.hardware_reset:
        reset_hardware()
        return

    serials = discover_serials()
    if not serials:
        print("No cameras found.  Connect at least one RealSense D415.")
        return

    out_root = Path(args.output_dir).resolve()
    cfg_path = Path(args.config_file).resolve()
    cfg = load_or_create_config(cfg_path)
    if args.num_imgs is not None:
        cfg["num_calibration_imgs"] = args.num_imgs

    # Charuco board and detection parameters
    aruco_params = aruco.DetectorParameters_create()
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    board = aruco.CharucoBoard_create(
        squaresX=args.charuco_cols,
        squaresY=args.charuco_rows,
        squareLength=args.square_length,
        markerLength=args.marker_length,
        dictionary=aruco_dict,
    )

    # Set up pipelines
    pipelines = []
    profiles = []
    for s in serials:
        plc = rs.pipeline()
        cfg_rs = rs.config()
        cfg_rs.enable_device(s)
        cfg_rs.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        cfg_rs.enable_stream(rs.stream.infrared, args.width, args.height, rs.format.y8, args.fps)
        profile = plc.start(cfg_rs)
        sensor = profile.get_device().first_depth_sensor()
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, float(args.exposure))
        sensor.set_option(rs.option.gain, float(args.gain))
        pipelines.append(plc)
        profiles.append(profile)

        # Record intrinsics into the JSON if missing
        ir_intr = rs.video_stream_profile(profile.get_stream(rs.stream.infrared)).get_intrinsics()
        color_intr = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()
        if s not in cfg["cams"]:
            cfg["cams"][s] = {"intrinsics": {}}
        cam_cfg = cfg["cams"][s]["intrinsics"]
        cam_cfg["img_size"] = [args.width, args.height]
        cam_cfg["focal_length"] = [color_intr.fx, color_intr.fy]
        cam_cfg["img_center"] = [color_intr.ppx, color_intr.ppy]
        cam_cfg["ir_focal_length"] = [ir_intr.fx, ir_intr.fy]
        cam_cfg["ir_img_center"] = [ir_intr.ppx, ir_intr.ppy]

        # Ensure folder structure exists
        (out_root / s / CALIB_DIR_NAME).mkdir(parents=True, exist_ok=True)
        (out_root / s / RECONST_DIR_NAME).mkdir(parents=True, exist_ok=True)

    save_config(cfg, cfg_path)

    print("Connected cameras:", ", ".join(serials))
    print("Press SPACE to capture.  ESC quits.")

    capture_count = 0
    while capture_count < cfg["num_calibration_imgs"]:
        frames = [plc.wait_for_frames() for plc in pipelines]
        ir_images = []
        valid_board = []
        for f in frames:
            ir = f.first(rs.stream.infrared)
            ir_np = np.asanyarray(ir.get_data())
            ir_images.append(ir_np)
            corners, ids, _ = aruco.detectMarkers(ir_np, aruco_dict, parameters=aruco_params)
            good = ids is not None and len(ids) >= args.threshold_charuco
            valid_board.append(good)

        stacked = np.hstack(ir_images) if len(ir_images) > 1 else ir_images[0]
        cv2.imshow("Infrared", stacked)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        if key == 32:  # SPACE
            if sum(valid_board) < args.min_views:
                print("Board not visible in enough cameras ({} < {}). Frame skipped.".format(sum(valid_board), args.min_views))
                continue
            capture_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for s, img in zip(serials, ir_images):
                fname = out_root / s / CALIB_DIR_NAME / f"image_{capture_count}.jpg"
                cv2.imwrite(str(fname), img)
            print(f"Saved set {capture_count}/{cfg['num_calibration_imgs']} (board visible in {sum(valid_board)} views)")

    cv2.destroyAllWindows()
    for plc in pipelines:
        plc.stop()
    print("Done.")

if __name__ == "__main__":
    main()
