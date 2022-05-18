#!/usr/bin/python3.8

from os import listdir
from PIL import Image
import matplotlib.pyplot as plt

def savelinkedimg(path):
	path_c = path + "compressed/"
	for file in listdir(path_c):
		c = Image.open(path_c + file)
		r = Image.open(path + "original/" + file)
		
		c = c.resize(r.size)
		
		linked = Image.new(c.mode, (r.width*2, r.height), (0, 0, 0))
		linked.paste(
			r
		)
		linked.paste(
			c, 
			(r.width, 0),
		)
		
		linked.save(path+"linked/"+file)
	

if __name__ == "__main__":
	for folder in listdir(r"/home/kernic/ISEN/License_plate_detection_and_image_enhancement/dataset/modified pix2pix/"):
		savelinkedimg(fr"/home/kernic/ISEN/License_plate_detection_and_image_enhancement/dataset/modified pix2pix/{folder}/")
