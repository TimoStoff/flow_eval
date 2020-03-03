#!/bin/bash
# inference.py can process a folder, this script will pass it all the folders in a parent folder
rosbag_dir="${1?"Usage: $0 <model.pth> <input/folder> <output/folder>"}"
flow_dir="${2?"Usage: $0 <model.pth> <input/folder> <output/folder>"}"
output_base_folder="${3?"Usage: $0 <model.pth> <input/folder> <output/folder>"}"
ours="${4?"Usage: $0 <model.pth> <input/folder> <output/folder> <ours>"}"
skip_frames=1
mkdir $output_base_folder
for bag in $rosbag_dir/*.bag; do
    sequence_name=$(basename "$bag")
	exp=$(echo $sequence_name | sed 's/\.bag//')
	cmd="rosrun ecnn_eval_flow ecnn_eval_flow $bag $flow_dir/$exp/yml $skip_frames /dvs/events /dvs/image_raw $output_base_folder/$exp $ours"
	echo $cmd
	$cmd
done
