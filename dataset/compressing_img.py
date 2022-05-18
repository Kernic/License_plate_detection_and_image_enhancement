#!/usr/bin/python3.8

import os 
from PIL import Image, ImageOps
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
        basewidth = int(float(img.size[0])/2)
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
		
		
        # 5. Compress every image and save it with a new name:
        img.save(r"../compressed/"+image, optimize=True, quality=5)

def resize_img(path):
    for file in os.listdir(path):
        img = Image.open(path+file)
        #shape = img.size
        img = img.resize((256, 256))
        im_flip = ImageOps.flip(img)
        im_mirror = ImageOps.mirror(img)

        #getting name of file
        name = '.'.join(file.split('.')[:-1])
        ext = file.split('.')[-1]

        img.save(path+file)
        im_flip.save(path+name+"_flip."+ext)
        im_mirror.save(path+name+"_mirror."+ext)
		
if __name__ == "__main__":
	for folder in os.listdir(r"/home/kernic/ISEN/License_plate_detection_and_image_enhancement/dataset/modified pix2pix"):
		PATH = fr"/home/kernic/ISEN/License_plate_detection_and_image_enhancement/dataset/modified pix2pix/{folder}/original/"
		resize_img(PATH)
		compress_images(PATH)
