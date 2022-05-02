# License plate detection and image enhancement

Project created for a school project, made by Corentin Le Gall and Alexandre Cadet

## Introduction :

This project aims for the detection of licence plate on lower quality images like CCTV or phone recorded videos. 
It is separated in 3 main parts : 
- License plate detection and extraction 
- Image enhancement 
- License plate reading and writing as text

## 1. License plate detection and extraction

For this part we used the neural network Yolo V.5 and trained it with the dataset containing 300 labeled images () and used noise data augmentation (10% of pixels, 3 per training images)
