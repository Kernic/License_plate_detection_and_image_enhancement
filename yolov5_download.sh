#!/bin/sh

# Downloading the yoloV5 GitHub project by ultralytics
git clone https://github.com/ultralytics/yolov5

# Installing python deps
cd yolov5
pip install -r requirements.txt
cd ..
