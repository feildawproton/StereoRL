import tensorflow as tf
from _map_image import indx_map_x
import numpy as np

def similarity_perpixel_grey(img, drawing):
	difference		= tf.math.subtract(img, drawing)
	abs_difference	= tf.math.abs(difference)
	similarity 		= tf.math.subtract(tf.ones(shape = drawing.shape), abs_difference)	

	reduced			= tf.reduce_sum(similarity, axis = -1)
	reduced 		= tf.reduce_sum(reduced, axis = -1)
	reduced 		= tf.reduce_sum(reduced)
	
	return similarity, reduced
	
def similarity(img, drawing):
	metric = tf.keras.metrics.CosineSimilarity(axis = -1) #along the color axis I think
	metric.update_state(img, drawing)
	return metric.result().numpy()
	
def similarity_perpixel(img, drawing):
	img_norm = tf.math.l2_normalize(img, axis = -1)
	drawing_norm = tf.math.l2_normalize(drawing, axis = -1)
	
	AmultB = tf.math.multiply(img_norm, drawing_norm)

	AB = tf.math.reduce_sum(AmultB, axis = -1)	
	
	AB_red = tf.math.reduce_sum(AB, axis = -1)
	AB_red = tf.math.reduce_sum(AB_red, axis = -1)
	
	return AB, AB_red
	
def leftfromright_error(y_true_both, y_pred_map):
	y_true_left = y_true_both[:,:,:,0:3]
	y_true_right = y_true_both[:,:,:,3:6]
	# let's let map be positive.  
	# therefore we will need to flip the sign here
	map = tf.math.multiply(y_pred_map, -1)
	
	map = tf.reduce_sum(map, axis = 0)
	y_true_right = tf.reduce_sum(y_true_right, axis = 0)
	y_true_left = tf.reduce_sum(y_true_left, axis = 0)
	
	map = map.numpy()
	map = map.astype(np.int32)
	y_true_right = y_true_right.numpy()
	y_true_left = y_true_left.numpy()
	
	print(map.shape)
	print("above is map shape")
	print(y_true_right.shape)
	print("above is right shape")
	print(y_true_left.shape)
	print("above is left shape")
	_ = input("hold up")
	
	left_prediction = indx_map_x(y_true_right, map)
	difference = tf.math.squared_difference(y_true_left, left_prediction)
	#need to get the dimensions back
	difference = tf.expand_dims(difference, axis = 0)
	print(difference.shape)
	_ = input("observe the difference")
	return difference
	
def classifier_leftfromright_similarity(y_true_both, y_pred_map):
	y_true_left = y_true_both[:,:,:,0:3]
	y_true_right = y_true_both[:,:,:,3:6]
	# let's let map be positive.  
	# therefore we will need to flip the sign here
	map = tf.math.multiply(y_pred_map, -1)
	left_prediction = indx_map_x(y_true_right, map)
	similarity, similarity_reduced = similarity_perpixel(y_true_left, left_prediction)
	return similarity_reduced
'''
# -- TESTING --
import os
import pickle
import matplotlib.pyplot as plt
import random

images_path = os.path.join("training_data", "downsampled_left_images.pkl")
with open(images_path, "rb") as f:
	left_imgs = pickle.load(f)
	
images_path = os.path.join("training_data", "downsampled_right_images.pkl")
with open(images_path, "rb") as f:
	right_imgs = pickle.load(f)
	
if len(left_imgs) != len(right_imgs):
	print("Data array lengths don't match.  BAIL!")
	exit()
	
plt.ion() #interactive mode
fig, axes = plt.subplots(nrows = 2, ncols = 2)

rnd = random.randint(0, len(left_imgs))

left_img = left_imgs[rnd]
right_img = right_imgs[rnd]

sim1, sim_red1 = similarity_perpixel(left_img, left_img)
sim2, sim_red2 = similarity_perpixel(left_img, right_img)

print(rnd)
print(sim_red1)
print(sim_red2)

axes[0][0].imshow(left_img)
axes[0][1].imshow(right_img)
axes[1][0].imshow(sim1, cmap = "gray")
axes[1][1].imshow(sim2, cmap = "gray")
plt.show()

_ = input("so what happened here?")
'''


