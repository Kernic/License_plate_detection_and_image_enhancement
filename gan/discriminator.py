# This File contain the class of the discriminator part of the Generative Adversarial Network

##########################################################
#                         Imports                        #
##########################################################
import tensorflow as tf

import os
import pathlib
import time
import datetime

import matplotlib.pyplot as plt

from common import *

def Discriminator():
	initializer = tf.random_normal_initializer(0., 0.02)

	inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
	tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

	x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

	down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
	down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
	down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

	zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
	conv = tf.keras.layers.Conv2D(
		512, 
		4, 
		strides=1,
		kernel_initializer=initializer,
		use_bias=False 
	)(zero_pad1)  # (batch_size, 31, 31, 512)

	batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

	leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

	zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

	last = tf.keras.layers.Conv2D(
		1, 
		4, 
		strides=1,
		kernel_initializer=initializer
	)(zero_pad2)  # (batch_size, 30, 30, 1)

	return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
	real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
	
	generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
	
	total_disc_loss = real_loss + generated_loss
	
	return total_disc_loss

def discriminator_describe():
	discriminator = Discriminator()
	tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64, to_file='./info/discriminator.png')

if __name__ == '__main__':
	discriminator_describe()

