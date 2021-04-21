import os
import pickle
import tensorflow as tf
import random

from _models import create_model
from _similarity_error import leftfromright_error, classifier_leftfromright_similarity

# -- LOAD DATA --
images_path = os.path.join("training_data", "downsampled_left_images.pkl")
with open(images_path, "rb") as f:
	left_imgs = pickle.load(f)
	
images_path = os.path.join("training_data", "downsampled_right_images.pkl")
with open(images_path, "rb") as f:
	right_imgs = pickle.load(f)
	
if len(left_imgs) != len(right_imgs):
	print("Data array lengths don't match.  BAIL!")
	exit()

combined_images = []
for indx, left_img in enumerate(left_imgs):
	right_img = right_imgs[indx]
	both = tf.concat([left_img, right_img], axis = -1)
	expanded = tf.expand_dims(both, axis = 0)
	combined_images.append(expanded)
	
#shuffle
zipped 					= list(zip(left_imgs, right_imgs))
random.shuffle(zipped)
left_imgs, right_imgs	= zip(*zipped)

TARGET_IMG_SHAPE		= (left_imgs[0].shape[0], left_imgs[0].shape[1], 3)
INPUT_SHAPE				= (TARGET_IMG_SHAPE[0], TARGET_IMG_SHAPE[1], 2 * TARGET_IMG_SHAPE[2])

BATCH_SIZE				= 64

LEARNING_RATE 			= 0.001	
KERNEL_DIM				= 3
KERNEL_DIMS				= (KERNEL_DIM,KERNEL_DIM)
FILTERS_PER_LAYER		= 32
	
NUM_EPOCHS				= 4
NUM_BATCHES				= int(len(left_imgs) / BATCH_SIZE)
	
model = create_model(INPUT_SHAPE, KERNEL_DIMS, 1, FILTERS_PER_LAYER)

model.compile(optimizer = tf.keras.optimizers.Adam(lr = LEARNING_RATE), loss = leftfromright_error, metrics = [leftfromright_error, classifier_leftfromright_similarity], run_eagerly = True)
	
checkpoint_path = './classifier_checkpoints'
os.makedirs(checkpoint_path, exist_ok=True)
model_path = os.path.join(checkpoint_path, 'model_epoch_{epoch:02d}_loss_{loss:.2f}_acc_{acc:.2f}_val_acc_{val_acc:.2f}.h5')

print(len(combined_images))
print(combined_images[random.randint(0,100)].shape)

history = model.fit(x = combined_images, y = combined_images, batch_size = BATCH_SIZE, epochs = NUM_EPOCHS, callbacks = [tf.keras.callbacks.ModelCheckpoint(model_path, monitor = "val_acc", save_best_only = True, verbose = 1)])

