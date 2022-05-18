##########################################################
#						 Imports						#
##########################################################
from tensorflow.keras import layers
import tensorflow as tf

import os
import pathlib
import time
import datetime

import matplotlib.pyplot as plt

def load(image_file):
	# Read and decode an image file to a uint8 tensor
	image = tf.io.read_file(image_file)
	image = tf.io.decode_jpeg(image)

	# Split each image tensor into two tensors:
	# - one with a real building facade image
	# - one with an architecture label image 
	w = tf.shape(image)[1]
	w = w // 2
	input_image = image[:, w:, :] /255
	real_image = image[:, :w, :] /255

	# Convert both images to float32 tensors
	input_image = tf.cast(input_image, tf.float32)
	real_image = tf.cast(real_image, tf.float32)

	return real_image, input_image

def downsample(filters, size, apply_batchnorm=True):
	initializer = tf.random_normal_initializer(0., 0.02)
	
	result = tf.keras.Sequential()
	result.add(
		tf.keras.layers.Conv2D(
			filters, 
			size, 
			strides=2, 
			padding='same',
		 	kernel_initializer=initializer, 
		 	use_bias=False
		)
	)
	
	if apply_batchnorm:
		result.add(tf.keras.layers.BatchNormalization())
	
	result.add(tf.keras.layers.LeakyReLU())
	
	return result
	
def upsample(filters, size, apply_dropout=False):
	initializer = tf.random_normal_initializer(0., 0.02)
	result = tf.keras.Sequential()
	result.add(
		tf.keras.layers.Conv2DTranspose(
			filters, 
			size, 
			strides=2,
			padding='same',
			kernel_initializer=initializer,
			use_bias=False
		)
	)
	
	result.add(tf.keras.layers.BatchNormalization())
	
	if apply_dropout:
		result.add(tf.keras.layers.Dropout(0.5))

	result.add(tf.keras.layers.ReLU())

	return result

def generate_images(model, test_input, tar):
	prediction = model(test_input, training=True)
	plt.figure(figsize=(15, 15))
	files = os.listdir(r"./info/trainingImages/")
	display_list = [test_input[0], tar[0], prediction[0]]
	title = ['Input Image', 'Ground Truth', 'Predicted Image']

	for i in range(3):
		plt.subplot(1, 3, i+1)
		plt.title(title[i])
		# Getting the pixel values in the [0, 1] range to plot.
		plt.imshow(display_list[i] * 0.5 + 0.5)
		plt.axis('off')
	plt.savefig(fr"./info/trainingImages/{len(files)}.png")


OUTPUT_CHANNELS = 3
LAMBDA = 100
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


