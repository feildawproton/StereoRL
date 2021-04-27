from numba import cuda, float32
import math
import numpy as np

#THREADS_PER_BLOCK = 64
THREADS_PER_BLOCK_2D = (8, 8)

# -- EXPECTED INPUTS ARE ROW MAJOR --
# -- INDICES ARE IN THE ORDER (Y, X) --

'''
plt.ion() #interactive mode
fig, axes = plt.subplots(nrows = 1, ncols = 2)
'''
# -- KERNEL FUNCTIONS --

@cuda.jit
def cuda_indx_map(source, target, map, y_limit, x_limit):
	y, x	= cuda.grid(2)
	y_rel 	= map[y][x][0]
	x_rel	= map[y][x][1]
	y_abs 	= y_rel + y				# Needs to be relative
	x_abs	= x_rel + x		
	#I KNOW IF STATEMENTS ARE BAD AND THIS ONE IS PARTICULARLY AGGREGIOUS
	#DON'T NEED TO CHECK Y_ABS AND X_ABS OUT OF BOUNDS?  RESULTS IN WRAPING
	if x < x_limit and y < y_limit and x_abs < x_limit and y_abs < y_limit and x_abs >= 0 and y_abs >= 0:
		target[y][x][0] = source[y_abs][x_abs][0]
		target[y][x][1] = source[y_abs][x_abs][1]
		target[y][x][2] = source[y_abs][x_abs][2]
'''	
@cuda.jit
def cuda_indx_map_grey(source, target, map, y_limit, x_limit):
	y, x	= cuda.grid(2)
	y_rel 	= map[y][x][0]
	x_rel	= map[y][x][1]
	y_abs 	= y_rel + y				# Needs to be relative
	x_abs	= x_rel + x		
	#I KNOW IF STATEMENTS ARE BAD AND THIS ONE IS PARTICULARLY AGGREGIOUS
	#DON'T NEED TO CHECK Y_ABS AND X_ABS OUT OF BOUNDS?  RESULTS IN WRAPING
	if x < x_limit and y < y_limit and x_abs < x_limit and y_abs < y_limit and x_abs >= 0 and y_abs >= 0:
		target[y][x] = source[y_abs][x_abs]
'''
@cuda.jit
def cuda_indx_map_x(source, target, map, y_limit, x_limit):
	y, x	= cuda.grid(2)
	x_rel 	= map[y][x][0]
	x_abs 	= x_rel + x				# Needs to be relative
	if x < x_limit and y < y_limit and x_abs < x_limit and x_abs >= 0:
		target[y][x][0] = source[y][x_abs][0]
		target[y][x][1] = source[y][x_abs][1]
		target[y][x][2] = source[y][x_abs][2]
'''
@cuda.jit
def cuda_indx_map_x_grey(source, target, map, y_limit, x_limit):
	y, x	= cuda.grid(2)
	x_rel 	= map[y][x][0]
	x_abs 	= x_rel + x				# Needs to be relative
	if x < x_limit and y < y_limit and x_abs < x_limit and x_abs >= 0:
		target[y][x] = source[y][x_abs]
	
@cuda.jit
def cuda_indx_map_x_grey_expanded(source, target, map, y_limit, x_limit):
	y, x	= cuda.grid(2)
	x_rel 	= map[y][x][0]
	x_abs 	= x_rel + x				# Needs to be relative
	if x < x_limit and y < y_limit and x_abs < x_limit and x_abs >= 0:
		target[y][x][0] = source[y][x_abs][0]
'''
def indx_map(source, target, map):
	if source.shape != target.shape:
		print("source and target sizes do not match (called from _map_image")
		return 1
	if map.ndim != 3:
		print("map dimensions must be 3: (y, x, (_y, _x))")
		return 1
	if map.dtype != np.int32:
		print("map needs to be of type np.int32")
		return 1
	if map.shape[2] != 2:
		print("expected two values (y,x) in map")
		return 1
	if source.shape[2] != 3 or target.shape[2] != 3:
		print("expected color images")
		return 1
	# -- Kernel Launch --
	blocks_per_grid_y 	= math.ceil(map.shape[0] / THREADS_PER_BLOCK_2D[0])
	blocks_per_grid_x 	= math.ceil(map.shape[1] / THREADS_PER_BLOCK_2D[1])
	blocks_per_grid 	= (blocks_per_grid_y, blocks_per_grid_x)
	cuda_indx_map[blocks_per_grid, THREADS_PER_BLOCK_2D](source, target, map, map.shape[0], map.shape[1])
	return 0
		
# -- FUNCTIONS TO CALL --
'''
def indx_map_grey(source, target, map):
	if source.shape != target.shape:
		print("source and target sizes do not match (called from _map_image")
		return 1
	if map.ndim != 3:
		print("map dimensions must be 3: (x, y, (_x, _y))")
		return 1
	if map.dtype != np.int32:
		print("map needs to be of type np.int32")
		return 1
	if map.shape[2] != 2:
		print("expected two values (y,x) in map")
		return 1
	if len(source.shape) > 2 or len(target.shape) > 2:
		print("expect grey images in the from y by x")
		return 1
	# -- Kernel Launch --
	blocks_per_grid_y 	= math.ceil(map.shape[0] / THREADS_PER_BLOCK_2D[0])
	blocks_per_grid_x 	= math.ceil(map.shape[1] / THREADS_PER_BLOCK_2D[1])
	blocks_per_grid 	= (blocks_per_grid_y, blocks_per_grid_x)
	cuda_indx_map_grey[blocks_per_grid, THREADS_PER_BLOCK_2D](source, target, map, map.shape[0], map.shape[1])
	return 0
'''
def indx_map_x(source, map):
	if map.ndim != 3:
		print("map dimensions must be 3: (x, y, (_x, _y))")
		return 1
	if map.dtype != np.int32:
		print("map needs to be of type np.int32")
		return 1
	if map.shape[2] != 1:
		print("this is just a one dimensional disparity map along x.  map should only have 1 value in last dimension")
		return 1
	if len(source.shape) != 3 and len(source.shape[2]) != 3:
		print("EXPECTED IMAGES WITH SHAPE Y BY X BY (r,g,b)")
		return 1
	# -- Need to copy --
	target 				= np.zeros(shape = source.shape, dtype = np.float32)
	this_source 		= np.zeros(shape = source.shape)
	this_source 		= np.copy(source)
	this_map 			= np.zeros(shape = map.shape)
	this_map 			= np.copy(map)

	# -- Kernel Launch --
	blocks_per_grid_y 	= math.ceil(map.shape[0] / THREADS_PER_BLOCK_2D[0])
	blocks_per_grid_x 	= math.ceil(map.shape[1] / THREADS_PER_BLOCK_2D[1])
	blocks_per_grid 	= (blocks_per_grid_y, blocks_per_grid_x)
	cuda_indx_map_x[blocks_per_grid, THREADS_PER_BLOCK_2D](this_source, target, this_map, this_map.shape[0], this_map.shape[1])

	return target
	
def indx_map_x_white(source, map):
	if map.ndim != 3:
		print("map dimensions must be 3: (x, y, (_x, _y))")
		return 1
	if map.dtype != np.int32:
		print("map needs to be of type np.int32")
		return 1
	if map.shape[2] != 1:
		print("this is just a one dimensional disparity map along x.  map should only have 1 value in last dimension")
		return 1
	if len(source.shape) != 3 and len(source.shape[2]) != 3:
		print("EXPECTED IMAGES WITH SHAPE Y BY X BY (r,g,b)")
		return 1
	# -- Need to copy --
	target 				= np.ones(shape = source.shape, dtype = np.float32)
	this_source 		= np.zeros(shape = source.shape)
	this_source 		= np.copy(source)
	this_map 			= np.zeros(shape = map.shape)
	this_map 			= np.copy(map)

	# -- Kernel Launch --
	blocks_per_grid_y 	= math.ceil(map.shape[0] / THREADS_PER_BLOCK_2D[0])
	blocks_per_grid_x 	= math.ceil(map.shape[1] / THREADS_PER_BLOCK_2D[1])
	blocks_per_grid 	= (blocks_per_grid_y, blocks_per_grid_x)
	cuda_indx_map_x[blocks_per_grid, THREADS_PER_BLOCK_2D](this_source, target, this_map, this_map.shape[0], this_map.shape[1])

	return target
'''
def indx_map_x_grey(source, map):
	if map.ndim != 3:
		print("map dimensions must be 3")
		return 1
	if map.dtype != np.int32:
		print("map needs to be of type np.int32")
		return 1
	if map.shape[2] != 1:
		print("this is just a one dimensional disparity map along x.  map should only have 1 value in last dimension")
		return 1
	if len(source.shape) < 3:
		print("EXPECTED GREY IMAGES WITH SHAPE Y BY X BY 1")
		return 1
	# -- Need to copy --
	target 				= np.zeros(shape = source.shape)
	this_source 		= np.zeros(shape = source.shape)
	this_source 		= np.copy(source)
	this_map 			= np.zeros(shape = map.shape)
	this_map 			= np.copy(map)
	# -- Kernel Launch --
	blocks_per_grid_y 	= math.ceil(map.shape[0] / THREADS_PER_BLOCK_2D[0])
	blocks_per_grid_x 	= math.ceil(map.shape[1] / THREADS_PER_BLOCK_2D[1])
	blocks_per_grid 	= (blocks_per_grid_y, blocks_per_grid_x)
	cuda_indx_map_x_grey_expanded[blocks_per_grid, THREADS_PER_BLOCK_2D](this_source, target, this_map, this_map.shape[0], this_map.shape[1])
	return target
'''

'''
import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

right_images_path = os.path.join("training_data", "downsampled_left_images.pkl")
with open(right_images_path, "rb") as f:
	right_images = pickle.load(f)

# -- TESTING X,Y --
right_image = right_images[15]
print(right_image.shape)

source = np.zeros(shape = right_image.shape)
source = np.copy(right_image)

target = np.zeros(shape = source.shape)

map_shape = (source.shape[0], source.shape[1], 2)
map = np.ones(shape = map_shape, dtype = np.int32)
#map = np.negative(map)

print(source.shape)
print(target.shape)
#print(map)

#_ = input("observe and dispare")

iter = 0
while iter < 10:
	indx_map(source, target, map)
	#axes[0].imshow(source, cmap = "gray")
	#axes[1].imshow(target, cmap = "gray")
	axes[0].imshow(source)
	axes[1].imshow(target)
	plt.show()
	_ = input("observe and dispare")
	source = target
	target = np.zeros(shape = source.shape)
	iter += 1
	
_ = input("let's do something else")

right_image = right_images[14]
print(right_image.shape)

source = np.zeros(shape = right_image.shape)
source = np.copy(right_image)

target = np.zeros(shape = source.shape)

map_shape = (source.shape[0], source.shape[1], 2)
map = np.ones(shape = map_shape, dtype = np.int32)
map = np.negative(map)

print(source.shape)
print(target.shape)

iter = 0
while iter < 10:
	indx_map(source, target, map)
	#axes[0].imshow(source, cmap = "gray")
	#axes[1].imshow(target, cmap = "gray")
	axes[0].imshow(source)
	axes[1].imshow(target)
	plt.show()
	_ = input("observe and dispare")
	source = target
	target = np.zeros(shape = source.shape)
	iter += 1
	
_ = input("let's do something else")

# -- TESTING RIGHT -> LEFT disparity --
right_image = right_images[10]
print(right_image.shape)

source = np.zeros(shape = right_image.shape)
source = np.copy(right_image)

target = np.zeros(shape = source.shape)

map_shape = (source.shape[0], source.shape[1], 1)
map = np.ones(shape = map_shape, dtype = np.int32)
#map = np.negative(map)

print(source.shape)
print(target.shape)

iter = 0
while iter < 10:
	indx_map_x(source, target, map)
	#axes[0].imshow(source, cmap = "gray")
	#axes[1].imshow(target, cmap = "gray")
	axes[0].imshow(source)
	axes[1].imshow(target)
	plt.show()
	_ = input("observe and dispare")
	source = target
	target = np.zeros(shape = source.shape)
	iter += 1
	
	
_ = input("let's do something else finaly")

right_image = right_images[213]
right_image = tf.math.reduce_mean(right_image, axis = -1)
print(right_image.shape)

source = np.zeros(shape = right_image.shape)
source = np.copy(right_image)

target = np.zeros(shape = source.shape)

map_shape = (source.shape[0], source.shape[1], 1)
map = np.ones(shape = map_shape, dtype = np.int32)
map = np.negative(map)

print(source.shape)
print(target.shape)

iter = 0
while iter < 10:
	indx_map_x_grey(source, target, map)
	axes[0].imshow(source, cmap = "gray")
	axes[1].imshow(target, cmap = "gray")
	#axes[0].imshow(source)
	#axes[1].imshow(target)
	plt.show()
	_ = input("observe and dispare")
	source = target
	target = np.zeros(shape = source.shape)
	iter += 1

_ = input("let's do something else finaly finally")



right_image = right_images[333]
right_image = tf.math.reduce_mean(right_image, axis = -1)
print(right_image.shape)

source = np.zeros(shape = right_image.shape)
source = np.copy(right_image)

#target = np.zeros(shape = source.shape)

map_shape = (source.shape[0], source.shape[1], 1)
map = np.ones(shape = map_shape, dtype = np.int32)

print(source.shape)

iter = 0
while iter < 10:
	target = indx_map_x_grey(source, map)
	axes[0].imshow(source, cmap = "gray")
	axes[1].imshow(target, cmap = "gray")
	#axes[0].imshow(source)
	#axes[1].imshow(target)
	plt.show()
	_ = input("observe and dispare")
	source = target
	target = np.zeros(shape = source.shape)
	iter += 1
	
_ = input("let's do something else")
'''
