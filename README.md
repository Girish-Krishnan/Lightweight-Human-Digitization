# 3D Human Modeling System

_Girish Krishnan, Benny Cai, Bang Du, Kunyao Chen, and Truong Nguyen_

A Python codebase for 3D reconstruction of human models from color (RGB) images and corresponding depth maps using an arbitrary number of Intel RealSense D415 cameras.

## Introduction
The __3D Human Modeling System__ is a fast and efficient solution for generating 3D models of humans from images and depth maps. It is designed to be easy to use and lightweight, making it suitable for a variety of applications where fast and accurate 3D reconstruction is required.

## Features
* 3D reconstruction using data from any arbitrary number of connected Intel RealSense D415 cameras.
* Option to use optimization algorithms such as point-to-point ICP and bundle adjustment.
* Outputs 3D models in __.ply__ file formats.

## Requirements
* Intel RealSense SDK 2.0 or higher
* Python 3.8 or higher
* Modules: matplotlib, numpy, open3d-python, opencv-python, pyrealsense2, scipy, utils

## Installation

1. Install the Intel RealSense SDK by following the instructions on the [official website](https://software.intel.com/en-us/realsense/sdk).

2. Clone the "Lightweight Human Digitization Repository":

```bash
git clone https://github.com/Girish-Krishnan/Lightweight-Human-Digitization.git
```

3. Install the required python modules:

```bash
cd Lightweight-Human-Digitization/Python
pip install -r requirements.txt
```

## Usage

### Capturing Images and Video

To capture images and depth maps from all currently connected Intel RealSense D415 cameras, run the following:

```bash
python capture.py
```

The program will identify all connected D415 cameras and capture an image from each camera simultaneously.

### Obtaining Calibration Images

To obtain the relative position and orientation between the cameras, the cameras need to be calibrated. This can be done using a __ChArUco__ calibration board.

To obtain images for calibration, run:

```bash
python charuco_images.py
```

Similar to *scalable_capturing.py*, a window will open showing live video feed from all connected cameras. If a ChArUco board is clearly visible through one or more cameras, the ArUco markers are detected and highlighted.

The total number of images needed for calibration depends on the number of cameras connected. Typically, around 20-30 images per camera pair is convenient. The total number of calibration images can be adjusted by editing the **configuration_parameters.json** file and changing the value of **num_calibration_images**.

Align the ChArUco board so that it is highlighted in at least two of the camera frames. Press the __spacebar__ to capture a calibration image. The terminal will display the number of images captured in total.

### Stereocalibration

To calibrate the cameras relative to one another, run:

```bash
python stereocalibrate.py
```

This will calibrate the cameras and store the calibration data in *configuration_parameters.json*

### Combining Point Clouds from Each Camera

To obtain a complete 3D human point cloud after calibration, run:

```bash
python combine_pcd.py
```

This will first show the individual point clouds for each camera (obtained using camera intrinsics) and finally show the complete point cloud obtained with the help of the extrinsic parameters.