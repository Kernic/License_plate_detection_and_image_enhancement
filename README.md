# License plate detection and image enhancement
<p align="center">
    <img src='https://github.com/Kernic/License_plate_detection_and_image_enhancement/blob/main/school_logo.jpg?raw=true'></img>
</p>
<p align="center">
Project created for a school project, made by <a href='https://github.com/Kernic'>Corentin Le Gall</a> and <a href='https://github.com/acadet22'>Alexandre Cadet</a>
</p>

## Introduction :

This project aims for the detection of licence plate on lower quality images like CCTV or phone recorded videos. 
It is separated in 3 main parts : 
- License plate detection and extraction 
- Image enhancement 
- License plate reading and writing as text

## 1. License plate detection and extraction

For this part we used the neural network Yolo V.5 and trained it with our dataset. 

The <a href='https://github.com/Kernic/License_plate_detection_and_image_enhancement/tree/main/dataset'>dataset</a> contained 300 labeled images that we splited on three part : train, valid, test (70, 20, 10). We the applied data augmentation on the train part of the dataset (noise generation 10% of pixels, 3 child image generated).

Next we used the <a href='https://github.com/ultralytics/yolov5'>Yolo V.5 github project</a> and trained the yolov5l on our data, optaining up to 98.11% of precision (details <a href='https://wandb.ai/kernic/train/runs/2f8ncml1?workspace=user-kernic'>here</a>)

<p align="center">
    <img src='https://github.com/Kernic/License_plate_detection_and_image_enhancement/blob/main/weights/Yolov5/infos/P_curve.png?raw=true' width='400' height='300'>
    <img src='https://github.com/Kernic/License_plate_detection_and_image_enhancement/blob/main/weights/Yolov5/infos/confusion_matrix.png?raw=true' width='400' height='300'>
</p>

## 2. Image enhancement

None

## 3. License plate reading and writing as text

None
