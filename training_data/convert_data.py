import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pickle

TARGET_SIZE = (84, 278)  #given in (height, width)
#GRAYSCALE = True
#COLOR_MODE = "grayscale"
GRAYSCALE = False
COLOR_MODE = "rgb"

LEFT_DIR = "image_2"
RIGHT_DIR = "image_3"

filenames_left = os.listdir(LEFT_DIR)
filenames_right = os.listdir(RIGHT_DIR)

images_left = []
images_right = []

for ndx, left_filename in enumerate(filenames_left):
	scale = 1. / 255.
	
	left_path = os.path.join(LEFT_DIR, left_filename)
	left_img = image.load_img(left_path, grayscale = GRAYSCALE, color_mode = COLOR_MODE, target_size = TARGET_SIZE)
	left_x = image.img_to_array(left_img)
	#left_x = tf.math.reduce_mean(left_x, axis = -1)
	left_x = tf.math.multiply(left_x, scale)
	
	right_filename = filenames_right[ndx]
	right_path = os.path.join(RIGHT_DIR, right_filename)
	right_img = image.load_img(right_path, grayscale = GRAYSCALE, color_mode = COLOR_MODE, target_size = TARGET_SIZE)
	right_x = image.img_to_array(right_img)
	#right_x = tf.math.reduce_mean(right_x, axis = -1)
	right_x = tf.math.multiply(right_x, scale)
	
	images_left.append(left_x)
	images_right.append(right_x)

left_name = "downsampled_left_images.pkl"
with open(left_name, "wb") as f:
	pickle.dump(images_left, f)
		
right_name = "downsampled_right_images.pkl"
with open(right_name, "wb") as f:
	pickle.dump(images_right, f)
		
print("done")
	
	
	
	