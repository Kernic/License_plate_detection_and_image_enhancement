#!/usr/bin/python3.8
import tensorflow as tf
import numpy as np
import os

def load(image_file):
	image = tf.io.read_file(image_file)
	image = tf.io.decode_jpeg(image)
	
	w = tf.shape(image)[1]
	w = w // 2
	
	input_image = image[:, :w, :] / 255
	shape = tf.shape(input_image)
	real_image = image[:, w:, :]  / 255
	
	input_image = tf.cast(input_image, tf.float32)
	real_image = tf.cast(real_image, tf.float32)
	
	return real_image, input_image
	
if __name__ == "__main__":
	PATH = r"../dataset/modified pix2pix/"
	dataset = tf.data.Dataset.list_files(PATH+"train/linked/*.jpg")
	dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
	
	for element in dataset:
		print(element)
		break
	
