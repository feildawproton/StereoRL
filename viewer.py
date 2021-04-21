import os
import pickle
import glob
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import time

path_names = glob.glob("./state_saves//*.pkl")

plt.ion() #interactive mode
fig, axes = plt.subplots(nrows = 2, ncols = 2)

for name in path_names:
	print("opening %s" % str(name))
	with open(name, "rb") as f:
		state = pickle.load(f)

	
	left = state[:,:,0:3]
	right = state[:,:,3:6]
	map = state[:,:,6]
	recreation = state[:,:,7:]
	
	axes[0,0].imshow(left)
	axes[0,1].imshow(right)
	axes[1,0].imshow(recreation)
	axes[1,1].imshow(map, cmap = "gray")
	
	plt.show()
	_ = input("Press something to continue")
	
