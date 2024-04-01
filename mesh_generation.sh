#!/bin/bash

# Default values
hardware_reset=false
data_reset=false
output_dir='./Capture_Data'
width=640
height=480
fps=60
warmup_frames=1000
num_captures=20
config_file='./configuration_parameters.json'
output_file='./point_cloud_combined.ply'
save_individual=false
visualize=false
odom_file='./odometry.log'
mesh_file='./mesh_combined.ply'

# Usage function
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --hardware_reset      Reset all connected cameras"
  echo "  --data_reset          Delete all captured data"
  echo "  --output_dir DIR      Output directory for captured data (default: './Capture_Data')"
  echo "  -w, --width           Width of captured images (default: 640)"
  echo "  -ht, --height         Height of captured images (default: 480)"
  echo "  -f, --fps             FPS of captured images (default: 60)"
  echo "  --warmup-frames       Number of frames to capture for warm-up (default: 1000)"
  echo "  -n, --num-captures    Number of images to capture (default: 20)"
  echo "  --config_file FILE    Configuration file (default: './configuration_parameters.json')"
  echo "  --output_file FILE    Output file for point cloud (default: './point_cloud_combined.ply')"
  echo "  --save_individual     Save individual images"
  echo "  --visualize           Visualize the output"
  echo "  --odom_file FILE      Odometry log file (default: './odometry.log')"
  echo "  --mesh_file FILE      Mesh file (default: './mesh_combined.ply')"
  exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --hardware_reset) hardware_reset=true ;;
    --data_reset) data_reset=true ;;
    --output_dir) output_dir="$2"; shift ;;
    -w|--width) width="$2"; shift ;;
    -ht|--height) height="$2"; shift ;;
    -f|--fps) fps="$2"; shift ;;
    --warmup-frames) warmup_frames="$2"; shift ;;
    -n|--num-captures) num_captures="$2"; shift ;;
    --config_file) config_file="$2"; shift ;;
    --output_file) output_file="$2"; shift ;;
    --save_individual) save_individual=true ;;
    --visualize) visualize=true ;;
    --odom_file) odom_file="$2"; shift ;;
    --mesh_file) mesh_file="$2"; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown parameter: $1"; usage ;;
  esac
  shift
done

# Your script's logic starts here
echo "Hardware reset: $hardware_reset"
echo "Data reset: $data_reset"
echo "Output directory: $output_dir"
echo "Width: $width"
echo "Height: $height"
echo "FPS: $fps"
echo "Warmup frames: $warmup_frames"
echo "Number of captures: $num_captures"
echo "Config file: $config_file"
echo "Output file: $output_file"
echo "Save individual: $save_individual"
echo "Visualize: $visualize"
echo "Odometry log file: $odom_file"
echo "Mesh file: $mesh_file"

set -e

if [ "$hardware_reset" = true ]; then
  python capture.py --hardware_reset
  exit 0
fi

python capture.py ${data_reset:+--data_reset} --output_dir "${output_dir}" -w $width -ht $height -f $fps --warmup-frames $warmup_frames -n $num_captures

python combine_pcd.py --config_file "${config_file}" --output_file "${output_file}" --data_dir "${output_dir}" ${save_individual:+--save_individual} ${visualize:+--visualize} --odom_file "${odom_file}" --mesh_file "${mesh_file}"

