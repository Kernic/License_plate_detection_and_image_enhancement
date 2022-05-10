#!/usr/bin/python3.8

import os 
from PIL import Image
import PIL
import os
import glob

def compress_images(directory=False, quality=10):
    # 1. If there is a directory then change into it, else perform the next operations inside of the 
    # current working directory:
    if directory:
        os.chdir(directory)
    
    # 2. Extract all of the .png and .jpeg files:
    files = os.listdir(directory)

    # 3. Extract all of the images:
    images = [file for file in files if file.endswith(('jpg', 'png'))]

    # 4. Loop over every image:
    for image in images:
        print(image)

        # 5. Open every image:
        img = Image.open(image)

        # 5. Compress every image and save it with a new name:
        img.save(r"../compressed/"+image, optimize=True, quality=quality)


if __name__ == "__main__":
	compress_images("/home/kernic/ISEN/ProjetIA/License_plate_detection_and_image_enhancement/dataset/modified pix2pix/valid/original/")
