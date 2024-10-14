#!/bin/bash

# Loop through numbers 1 to 100
for x in {1..100}
do
    # Execute the command with the current value of x
    python combine_pcd.py --data_dir DATASET/Video --mesh_file ./video_frame_${x}.mesh --frame_number ${x}
done
