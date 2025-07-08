# Lightweight Human Digitization

Python utilities for capturing, calibrating, and reconstructing three‑dimensional human models with several Intel RealSense D415 cameras.

## Table of Contents
- [Lightweight Human Digitization](#lightweight-human-digitization)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Directory Layout](#directory-layout)
  - [Workflow Overview](#workflow-overview)
  - [Step 1 Calibration Images](#step1calibration-images)
  - [Step 2 Stereo Calibration](#step2stereo-calibration)
  - [Step 3 Capturing New Scans](#step3capturing-new-scans)
    - [Using the GUI](#using-the-gui)
    - [Using the Command Line](#using-the-command-line)
  - [Step 4 Point Cloud Fusion and Mesh Generation](#step4point-cloud-fusion-and-mesh-generation)
  - [Recording and Converting bag Files](#recording-and-converting-bag-files)
  - [Shell Helper mesh\_generation.sh](#shell-helpermesh_generationsh)
  - [Troubleshooting](#troubleshooting)

---

## Quick Start
```bash
# clone the repo
git clone https://github.com/Girish-Krishnan/Lightweight-Human-Digitization.git
cd Lightweight-Human-Digitization

# install Python packages (use a virtual environment if possible)
pip install -r requirements.txt

# first capture calibration data and solve extrinsics
python charuco_images.py --output_dir ./Capture_Data
python stereocalibrate.py --data_dir ./Capture_Data

# then launch the graphical interface for everyday scans
# (you can just use the default parameters and hit "Submit")
python gui.py
```
The GUI completes capture, fusion, and mesh export after calibration has been saved.

## Installation
* Intel RealSense SDK 2.0 or later with up‑to‑date firmware on each camera
* Python 3.8 or later
* `pip install -r requirements.txt` inside the project folder

## Directory Layout
```
Lightweight-Human-Digitization/
├── capture.py              synchronous RGB‑D capture
├── charuco_images.py       ChArUco calibration frame capture
├── stereocalibrate.py      extrinsic solver for any number of cameras
├── combine_pcd.py          point cloud merging and mesh export
├── gui.py                  graphical front end
├── process_bag.py          convert bag recordings to RGB‑D frames
├── record_bag.py           simple RealSense recorder
├── mesh_generation.sh      helper that runs capture and fusion in sequence
├── requirements.txt
└── README.md
```
The first run of any capture script creates `Capture_Data/` with one subfolder per camera.

## Workflow Overview
1. **Calibration images**—grab a series of ChArUco snapshots with `charuco_images.py`.
2. **Stereo calibration**—run `stereocalibrate.py` once.  The script writes extrinsics to `configuration_parameters.json`.
3. **Capture**—use the GUI `gui.py` or call `capture.py` from the shell.
4. **Merge point clouds and build a watertight mesh**—`combine_pcd.py` does this automatically.  The GUI calls it for you.

Repeat the calibration stages only when cameras have been moved.

---

## Step 1 Calibration Images
`charuco_images.py` shows a live infrared feed from every camera.  A frame is accepted when the board is visible in at least `--min_views` cameras, default one.  Press **space** to store a set.

```bash
python charuco_images.py \
    --output_dir ./Capture_Data \
    --charuco_rows 9 \
    --charuco_cols 12 \
    --square_length 0.060 \
    --marker_length 0.044 \
    --num_imgs 30 \
    --min_views 2          # require the board in two cameras
```
Flags of interest
* `--hardware_reset` resets every attached sensor then exits.
* `--output_dir` chooses where images are written.
* Camera controls `--width` `--height` `--fps` `--exposure` `--gain`.
* `--threshold_charuco` sets the number of visible marker ids needed inside each accepted frame.

The script builds `configuration_parameters.json` from scratch if it does not exist and appends intrinsics for any newly discovered cameras.  Nothing else in the file is changed.

## Step 2 Stereo Calibration
```bash
python stereocalibrate.py \
    --config_file ./configuration_parameters.json \
    --data_dir     ./Capture_Data \
    --bundle_adjust              # optional nonlinear refinement
```
After success, `configuration_parameters.json` gains camera–camera rotation and translation blocks used by later processing.

## Step 3 Capturing New Scans
### Using the GUI
```bash
python gui.py
```
The window lets you pick storage paths, resolution, frame rate, warm‑up count, and capture count.  Browse for the calibration JSON, choose point‑cloud and mesh file names, and click **Submit**.

### Using the Command Line
```bash
python capture.py \
    --output_dir ./Capture_Data \
    -w 640 -ht 480 -f 60 \
    --warmup-frames 1000 \
    -n 20
```
Important flags
* `--hardware_reset` resets hardware and exits.
* `--data_reset` removes existing RGB‑D files under `--output_dir`.
* `--warmup-frames` discards early frames to allow sensor gain and exposure to stabilise.
* `-n` sets the number of captures.  When the value is greater than one the files are numbered `image_1.jpg`, `depth_map_1.npy`, and so on.

Each camera writes into `<serial>/reconstruction_images/`.

---

## Step 4 Point Cloud Fusion and Mesh Generation
```bash
python combine_pcd.py \
    --config_file ./configuration_parameters.json \
    --data_dir     ./Capture_Data \
    --output_file  ./point_cloud_combined.ply \
    --mesh_file    ./mesh_combined.ply \
    --odom_file    ./odometry.log \
    --frame_number 1 \
    --save_individual   # keep per‑camera clouds
```
Extra switches:
* `--save_individual` stores one PLY per camera before fusion.

The script estimates normals, runs Poisson reconstruction with depth eight, smooths, removes small clusters, re‑colors vertices, then saves both cloud and mesh.  Increase the Poisson `depth` value inside the script when you need finer detail at the cost of longer processing time.

---

## Recording and Converting bag Files
**Record** when real‑time capture is not possible.
```bash
python record_bag.py            # five seconds from every camera
```
**Convert** to RGB‑D frames.
```bash
python process_bag.py           # writes frames to Capture_Data
```
Edit the `bag_files` list inside `process_bag.py` for your own recordings.

---

## Shell Helper mesh_generation.sh
One call that performs capture and fusion.
```bash
bash mesh_generation.sh \
    --output_dir ./Capture_Data \
    -w 640 -ht 480 -f 60 \
    --warmup-frames 1000 \
    -n 20 \
    --output_file ./point_cloud_combined.ply \
    --mesh_file   ./mesh_combined.ply
```
All flags mirror those of the Python scripts.  Any flag left out falls back to the default inside each script.

---

## Troubleshooting
* **No cameras detected**—run `rs-enumerate-devices` and update firmware if needed.
* **Depth frames are all zeros**—remove the factory lens cover or increase projector power.
* **Calibration fails**—ensure each pair of cameras shares at least twenty common markers.  Lower `--threshold_charuco` when working with fewer markers.
* **Mesh has holes**—capture at higher resolution, bring cameras closer, or raise the Poisson depth value in `combine_pcd.py`.
