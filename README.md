# Lightweight Human Digitization

Python tools for multi‑camera capture, calibration, and 3D reconstruction of a human subject with Intel RealSense D415 cameras.

## Table of Contents

- [Lightweight Human Digitization](#lightweight-human-digitization)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quickstart)
  - [Installation](#installation)
  - [Directory Layout](#directorylayout)
  - [Workflow Overview](#workflowoverview)
  - [Step 1 – Calibration Images](#step1calibration-images)
  - [Step 2 – Stereo Calibration](#step2stereo-calibration)
  - [Step 3 – Capturing New Scans](#step3capturing-new-scans)
    - [Using the GUI](#using-the-gui)
    - [Using the Command Line](#using-the-command-line)
  - [Step 4 – Point‑Cloud Fusion and Mesh Generation](#step4pointcloud-fusionandmesh-generation)
  - [Recording and Converting *.bag* Files](#recording-and-convertingbagfiles)
  - [Shell Helper: `mesh_generation.sh`](#shell-helper-mesh_generationsh)
  - [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# clone the repo
git clone https://github.com/Girish-Krishnan/Lightweight‑Human‑Digitization.git
cd Lightweight‑Human‑Digitization

# install Python packages (v‑env recommended)
pip install -r requirements.txt

# launch the graphical interface
python gui.py
```

The GUI handles capture and reconstruction in one click after calibration has been saved.

## Installation

* Intel RealSense SDK 2.0 or later, with firmware updated on each camera
* Python 3.8 or later
* Run `pip install -r requirements.txt` inside the project folder

## Directory Layout

```
Lightweight‑Human‑Digitization/
├── capture.py              synchronous RGB‑D capture
├── charuco_images.py       capture of ChArUco calibration frames
├── stereocalibrate.py      multi‑camera extrinsic solver
├── combine_pcd.py          point‑cloud merging and mesh output
├── gui.py                  graphical front‑end
├── process_bag.py          convert *.bag* recordings to RGB‑D frames
├── record_bag.py           quick multi‑camera recorder
├── mesh_generation.sh      end‑to‑end helper script
├── requirements.txt
└── README.md
```

The first run of each capture script creates a `Capture_Data/` folder that stores per‑camera images and depth maps.

## Workflow Overview

1. **Calibration images** – take ChArUco snapshots with `charuco_images.py`.
2. **Stereo calibration** – run `stereocalibrate.py` once; this writes extrinsics to `configuration_parameters.json`.
3. **Capture** – either the GUI (`gui.py`) or the command line (`capture.py`).
4. **Merge point clouds and build a watertight mesh** – `combine_pcd.py`.

Calibration needs to be repeated only when cameras move.

---

## Step 1 – Calibration Images

Ensure the ChArUco board is in view of at least two cameras before each shot; press **space** to store a frame.

```bash
python charuco_images.py \
    --output_dir ./Capture_Data \
    --charuco_rows 9 \
    --charuco_cols 12 \
    --square_length 0.060 \
    --marker_length 0.044 \
    --num_calibration_imgs 30
```

Flags:

* `--hardware_reset` reset every connected RealSense sensor and exit.
* `--output_dir` write images to this folder.
* Camera settings: `--width` `--height` `--fps` `--exposure` `--gain`.
* `--threshold_charuco` minimum common markers needed per frame.

All options are optional; defaults are chosen for D415 sensors at **640×480 @ 30 fps**.

## Step 2 – Stereo Calibration

```bash
python stereocalibrate.py \
    --config_file ./configuration_parameters.json \
    --data_dir     ./Capture_Data \
    --bundle_adjust              # refine with non‑linear optimization (optional)
```

After success, `configuration_parameters.json` contains a block like:

```jsonc
"cams": {
  "828612060381": {
     "intrinsics": { ... },
     "839112060979": {
        "rotation": [...],
        "translation": [...]
     }
  },
  ...
}
```

This file is read by later stages.

## Step 3 – Capturing New Scans

### Using the GUI

```bash
python gui.py
```

The window lets you:

* reset cameras,
* pick the storage folder,
* choose resolution, frame rate, warm‑up count, number of captures,
* browse for the calibration JSON, and output filenames,
* toggle saving of individual point clouds and mesh visualisation.
  Click **Submit** to record, post‑process, and build the mesh. Status messages appear in the terminal and dialog boxes.

### Using the Command Line

```bash
python capture.py \
    --output_dir ./Capture_Data \
    -w 640 -ht 480 -f 60 \
    --warmup-frames 1000 \
    -n 20
```

Main flags:

* `--hardware_reset`   reset hardware then quit.
* `--data_reset`       remove all files in `--output_dir` before capture.
* `--output_dir`       root folder for RGB‑D images.
* `-w` `-ht` `-f`      width, height, and fps.
* `--warmup-frames`    discard this many synced frames before the first saved shot.
* `-n`                 number of synced RGB‑D pairs to store. When `‑n` > 1, files are numbered: `image_1.jpg`, `depth_map_1.npy`, …

The script writes one sub‑folder per camera: `serial/reconstruction_images/`.

---

## Step 4 – Point‑Cloud Fusion and Mesh Generation

```bash
python combine_pcd.py \
    --config_file ./configuration_parameters.json \
    --data_dir     ./Capture_Data \
    --output_file  ./point_cloud_combined.ply \
    --mesh_file    ./mesh_combined.ply \
    --odom_file    ./odometry.log \
    --frame_number 1 \
    --save_individual   # optional, keep per‑camera PLY files
```

Extra switches:

* `--visualize`        open an Open3D viewer for the merged cloud.
* `--save_individual`  export a PLY file for every camera before fusion.

The script estimates normals, runs Poisson surface reconstruction, smooths, removes small clusters, colors vertices, then saves both point cloud and mesh.

---

## Recording and Converting *.bag* Files

*Record only when real‑time capture is impractical.*

**Record**

```bash
python record_bag.py        # five‑second clip from each attached sensor
```

**Convert**

```bash
python process_bag.py       # writes RGB and depth frames to Capture_Data
```

Edit the `bag_files` array inside `process_bag.py` or adapt the script.

---

## Shell Helper: `mesh_generation.sh`

Single command that wraps capture and fusion:

```bash
bash mesh_generation.sh \
    --output_dir ./Capture_Data \
    -w 640 -ht 480 -f 60 \
    --warmup-frames 1000 \
    -n 20 \
    --output_file ./point_cloud_combined.ply \
    --mesh_file   ./mesh_combined.ply
```

Flags mirror the Python scripts; unset flags fall back to sensible defaults.

---

## Troubleshooting

* **No cameras detected** – check `rs-enumerate-devices`; update firmware.
* **Depth frames all zero** – remove camera cover glass or raise projector power.
* **Calibration fails** – verify at least twenty ChArUco markers are common across every pair. Increase `--threshold_charuco` only after successful runs.
* **Mesh has holes** – try higher resolution capture, add cameras on the missing side, or raise `--frame_number` for a clearer pose.
