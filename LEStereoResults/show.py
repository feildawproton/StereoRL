import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
#from tensorflow.keras.preprocessing import image

from _map_image import indx_map_x

def disparity_x_for_drawing(map):
	fmap = map.astype(np.float32)
	const = - 1. / 255.0
	
	fmap = fmap * const 
	
	#print(fmap)
	return fmap

plt.ion() #interactive mode
fig, axes = plt.subplots(nrows = 2, ncols = 2)

testing_left_path = os.path.join("testing", "image_2")
testing_right_path = os.path.join("testing", "image_3")

left_image_names = os.listdir(testing_left_path)
right_image_names = os.listdir(testing_right_path)
disp_names = os.listdir("disp_0")


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
	
	this_map = np.zeros(shape = map.shape)
	this_map = this_map.astype(np.int32)
	
	#print(np.max(image))
	#print(np.min(map))
	print(map.shape)
	#print(this_map)
	
	left_img_name = left_image_names[indx]
	left_img_path = os.path.join(testing_left_path, left_img_name)
	left_image = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
	
	right_img_name = right_image_names[indx]
	right_img_path = os.path.join(testing_right_path, right_img_name)
	right_image = cv2.imread(right_img_path, cv2.IMREAD_COLOR)
	
	recreation = indx_map_x(right_image, map)
	print(recreation.shape)
	print(left_image.shape)
	print(right_image.shape)
	axes[0,0].imshow(left_image)
	axes[0,1].imshow(right_image)
	axes[1,0].imshow(recreation / 256)
	axes[1,1].imshow(disparity_x_for_drawing(map), cmap = "gray")
	plt.show()

	_ = input("hold up")
	

