import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from _similarity_error import similarity_perpixel
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from _map_image import indx_map_x

def disparity_x_for_drawing(map):
	fmap = map.astype(np.float32)
	const = - 1. / 255.0
	
	fmap = fmap * const 
	
	#print(fmap)
	return fmap

testing_left_path = os.path.join("testing", "image_2")
testing_right_path = os.path.join("testing", "image_3")

left_image_names = os.listdir(testing_left_path)
right_image_names = os.listdir(testing_right_path)
disp_names = os.listdir("disp_0")

acc_array = np.zeros(shape = len(left_image_names))

for indx, name in enumerate(disp_names):
	full_path = os.path.join("disp_0", name)
	'''
	disp_image = image.load_img(full_path, grayscale = True, color_mode = "grayscale")
	disp_array = image.img_to_array(disp_image)
	disp_array = tf.math.multiply(disp_array, (-1./255))
	print(disp_array)
	_ = input("hold up")
	'''
	image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
	map = image.astype(np.int32)
	map = map * -1
	map = np.expand_dims(map, axis = -1)
	#left_image = cv2.resize(left_image, (278, 84))

	#this_map = np.zeros(shape = map.shape)
	#this_map = this_map.astype(np.int32)
	
	#print(np.max(image))
	#print(np.min(map))
	#print(map.shape)
	#print(this_map)
	
	left_img_name = left_image_names[indx]
	left_img_path = os.path.join(testing_left_path, left_img_name)
	left_image = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
	left_image = left_image.astype(np.float32)
	left_image = left_image / 255.0
	
	
	right_img_name = right_image_names[indx]
	right_img_path = os.path.join(testing_right_path, right_img_name)
	right_image = cv2.imread(right_img_path, cv2.IMREAD_COLOR)
	right_image = right_image.astype(np.float32)
	right_image = right_image / 255.0
	
	recreation = indx_map_x(right_image, map)
	#recreation = recreation.astype(np.uint8)
	
	
	
	left_image = cv2.resize(left_image, (278, 84))
	recreation = cv2.resize(recreation, (278, 84))
	
	# -- INSERT WHITE HERE -- 
	equal2condition = tf.math.equal(recreation, 0.0)
	recreation_oned = tf.where(equal2condition, 1.0, recreation)
	
	sim, sim_red = similarity_perpixel(left_image, recreation)
	#sim, sim_red = similarity_perpixel(left_image, right_image)
	print(sim_red.numpy())

	val = sim_red / (left_image.shape[0] * left_image.shape[1])
	val = val.numpy()
	acc_array[indx] = val

print(np.mean(acc_array))
print(np.std(acc_array))

