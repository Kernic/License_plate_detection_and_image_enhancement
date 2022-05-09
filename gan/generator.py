# This File contain the class of the generator part of the Generative Adversarial Network

##########################################################
#                         Imports                        #
##########################################################
from tensorflow.keras import layers
import tensorflow as tf

import os
import pathlib
import time
import datetime

import matplotlib.pyplot as plt

from common import *

def Generator():
	inputs = tf.keras.layers.Input(shape=[None, None, 3])

	down_stack = [
		downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
		downsample(128, 4),  # (batch_size, 64, 64, 128)
		downsample(256, 4),  # (batch_size, 32, 32, 256)
		downsample(512, 4),  # (batch_size, 16, 16, 512)
		downsample(512, 4),  # (batch_size, 8, 8, 512)
		downsample(512, 4),  # (batch_size, 4, 4, 512)
		downsample(512, 4),  # (batch_size, 2, 2, 512)
		downsample(512, 4),  # (batch_size, 1, 1, 512)
  	]

	up_stack = [
		upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
		upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
		upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
		upsample(512, 4),  # (batch_size, 16, 16, 1024)
		upsample(256, 4),  # (batch_size, 32, 32, 512)
		upsample(128, 4),  # (batch_size, 64, 64, 256)
		upsample(64, 4),  # (batch_size, 128, 128, 128)
  	]

	initializer = tf.random_normal_initializer(0., 0.02)
	last = tf.keras.layers.Conv2DTranspose(
		3, 
		4,
		strides=2,
		padding='same',
		kernel_initializer=initializer,
		activation='tanh'
	)

	x = layers.Resizing(256, 256)(inputs)

	# Downsampling through the model
	skips = []
	for down in down_stack:
		x = down(x)
		skips.append(x)

	skips = reversed(skips[:-1])

	# Upsampling and establishing the skip connections
	for up, skip in zip(up_stack, skips):
		x = up(x)
		x = tf.keras.layers.Concatenate()([x, skip])
	
	x = upsample(64, 4)(x)
	
	x = last(x)

	return tf.keras.Model(inputs=inputs, outputs=x)
	

def generator_loss(disc_generated_output, gen_output, target):
	loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

	# Mean absolute error
	l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

	total_gen_loss = gan_loss + (100 * l1_loss)

	return total_gen_loss, gan_loss, l1_loss

if __name__ == "__main__":

	inp, re = load('./test.jpeg')
	generator = Generator()
	tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64, to_file='./generator.png')
	
	gen_output = generator(inp[tf.newaxis, ...], training=False)
	plt.imshow(gen_output[0, ...])
	plt.show()
