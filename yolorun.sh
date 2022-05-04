#!/bin/sh

FILE = ./yolov5
# cheking if the yolov5 is downloaded in the root file of the project
if [! -f "$FILE"]; then
    echo "Yolo v5 not downloaded, please run the yolov5_download.sh to fix this probleme."
    exit 1
fi

# runing the python file for image detections
# TBW
