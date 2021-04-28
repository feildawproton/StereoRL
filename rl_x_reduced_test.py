import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import random
import os
import numpy as np
import math
import matplotlib.pyplot as plt

from _models import create_model
from _map_image import indx_map_x
from _similarity_error import similarity_perpixel
from _update_map import update_map_x
	
def disparity_x_for_drawing(map):
	fmap = tf.cast(map, dtype = tf.float32)
	fmap = tf.math.multiply(fmap, (-1.0 / 255.0))
	
	#print(fmap)
	return fmap
	
def env_state(reference, source, map, recreation):
	return tf.concat([reference, source, disparity_x_for_drawing(map), recreation], axis = -1)
	
def save_state(next_state, image_ndx, epoch_iter, step, reward):
	state_name = "best_state_image" + str(image_ndx) + "_epoch" + str(epoch_iter) + "_step" + str(step) + "_score" + str(reward) + ".pkl"
	state_path = os.path.join("state_saves", state_name)
	with open(state_path, "wb") as f:
		pickle.dump(next_state, f)

# -- LOAD DATA --
images_path = os.path.join("testing", "downsampled_left_images.pkl")
with open(images_path, "rb") as f:
	left_imgs = pickle.load(f)
	
images_path = os.path.join("testing", "downsampled_right_images.pkl")
with open(images_path, "rb") as f:
	right_imgs = pickle.load(f)
	
if len(left_imgs) != len(right_imgs):
	print("Data array lengths don't match.  BAIL!")
	exit()
	
#shuffle
zipped 					= list(zip(left_imgs, right_imgs))
random.shuffle(zipped)
left_imgs, right_imgs	= zip(*zipped)
	
# -- SET CONST --
TARGET_IMG_SHAPE		= (left_imgs[0].shape[0], left_imgs[0].shape[1], 3)
MAP_SHAPE				= (TARGET_IMG_SHAPE[0], TARGET_IMG_SHAPE[1], 1)
NUM_IMAMGE_INPUTS		= 3
STATE_SHAPE 			= (TARGET_IMG_SHAPE[0], TARGET_IMG_SHAPE[1], NUM_IMAMGE_INPUTS * TARGET_IMG_SHAPE[2] + MAP_SHAPE[2])
NUM_ACTIONS 			= 3																					# down, nothing, and up
OUTPUT_SHAPE 			= (TARGET_IMG_SHAPE[0], TARGET_IMG_SHAPE[1], NUM_ACTIONS)	
GAMMA					= 0.99																				# discount factor for predicted future rewards


UPDATE_TARGET_NETWORK	= len(left_imgs) / 10																# how often to update the target model
LOSS_FUNCTION 			= keras.losses.Huber()																# seems like what is typically used.  was used by pixelrl
EPS_START				= 0.9  																				# used for decaying probability of random action
EPS_END					= 0.05
PATIENCE_COUNT			= 5
PATIENCE_TEST			= 1.0

MAX_STEPS_PER_IMAGE		= 512
EPS_DECAY				= MAX_STEPS_PER_IMAGE / 4
UPDATE_AFTER_ACTIONS	= 8																					# how often to train the model.  used 4 from examples.  borrowed this from keras rl examples
BATCH_SIZE				= 32
NUM_EPOCHS				= 1																					# the number of times we will go through the whole 

# -- HYPERPARAMETERS TO VARY -- 
LEARNING_RATE 			= 0.001																				# was 0.00025 in the keras reference 
optimizer				= keras.optimizers.Adam(learning_rate = LEARNING_RATE, clipnorm = 1.0)

# -- FOR VIZ --
'''
plt.ion() #interactive mode
fig, axes = plt.subplots(nrows = 2, ncols = 2)
'''

# -- CREATE MODELS --	
model_path 				= os.path.join("models", "5epoch_4Inputs_5LayerNormed_3KERNEL_32FILTERS_0.001LR_32batch")
model					= tf.keras.models.load_model(model_path)														# action probabilities -> environment -> calculated rewards
target_model			= tf.keras.models.load_model(model_path)														# reward predictions.  used with model to calculate losses
target_model.summary()

# -- EPOCH LOOPS --
patience_tested	= 0
epoch_iter		= 0




while epoch_iter < NUM_EPOCHS:
	
	print("LEARNING_RATE: %f, BATCH_SIZE: %i, EPOCH: %i" % (LEARNING_RATE, BATCH_SIZE, epoch_iter))
	
	# -- TESTING --
	best_recreation_values = np.zeros(shape = len(left_imgs))
	
	# -- LOOP THROUGH THE IMAGES
	for image_ndx, left_ref in enumerate(left_imgs):
	
	
		n_rand_actions 			= 0		# for printing
		n_model_actions 		= 0		# ""
		n_model_backprops 		= 0		# ""
		n_target_updates		= 0		# ""
		#episode_reward 		= 0		# could be used for early stopping
		patience_tested			= 0		# num times episode_final_reward_reducxed < 1.0 
		final_reward_reducxed	= 0		# used to test patience
		
		
		action_history 			= []
		state_history 			= []	# before applying an action
		state_next_history		= []	# after applying an action
		reward_history 			= []
		map_history 			= [] 	# updata a map from actions and apply that to source
		#source_history			= []
		
		
		print("working on image %i out of %i" % (image_ndx, len(left_imgs)))
		
		# LEFT IMAGE IS CONST, 
		# RIGHT IMAGE IS SOURCE FOR RECREATION OF LEFT IMAGE USING A MAP.
		# MAP(RIGHT_IMAGE) APPROX LEFT IS THE RECREATION
		# RECREATE THE RECREATION EVERY STEP.  DO NOT USE IT AS A SOURCE	
		right_src 				= right_imgs[image_ndx]
		
		recreation 				= np.zeros(shape = right_src.shape)
		recreation				= np.copy(right_src)

		map 					= np.zeros(shape = MAP_SHAPE, dtype = np.int32)
		state 					= env_state(left_ref, right_src, map, recreation)
		
		reward, reward_reduced	= similarity_perpixel(left_ref, right_src)
		best_reward				= reward_reduced.numpy()
		print(reward_reduced)
		print("starting reward")
		best_state = env_state(left_ref, right_src, map, recreation)
		fount_at_step = 0
		
		# -- ITERATE OVER A SINGLE IMAGE --
		for step in range(1, MAX_STEPS_PER_IMAGE):
			smpl			= random.random()
			tot_step		= step + (image_ndx * MAX_STEPS_PER_IMAGE) + (epoch_iter * len(left_imgs))
			eps_thresh 		= EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step / EPS_DECAY)
			
			state_tensor	= tf.convert_to_tensor(state)
			
			# -- RANDOM OR MODEL ACTION --
			if smpl > eps_thresh:
				n_model_actions 		+= 1
				state_tensor 			= tf.expand_dims(state_tensor, axis = 0)
				action_probabilities	= model(state_tensor, training = False)
				action_probabilities	= tf.reduce_sum(action_probabilities, axis = 0)		#because there is a leading dimension in the returned tensor [batches, y, x, actions].  batches = 1
				action_numbers 			= tf.math.argmax(action_probabilities, axis = -1)
				
			else:
				n_rand_actions		+= 1
				action				= random.randint(0, NUM_ACTIONS - 1)
				action_numbers		= tf.fill([OUTPUT_SHAPE[0], OUTPUT_SHAPE[1]], action)
				# we could do this:  action_probabilities	= tf.random.normal(OUTPUT_SHAPE, mean = 0.5, stddev = 0.25)
				# but that would scrable the image and I think make learning harder
				
			# -- UPDATE MAP AND APPLY TO RECREATION --
			map 				= update_map_x(map, action_numbers, MAP_SHAPE)
			next_recreation		= indx_map_x(right_src, map)
			next_state			= env_state(left_ref, right_src, map, next_recreation)
			
			# -- CALC REWARD --
			reward, reward_reduced = similarity_perpixel(left_ref, next_recreation)
						
			# -- SAVE STATE_NEXT FOR VIEWING IF IT IS GOOD
			reward_reduced_value = reward_reduced.numpy()
			if reward_reduced_value > best_reward:
				best_reward = reward_reduced_value
				best_state = next_state
				fount_at_step = step
			
			# -- UPDATE RECORDS --
			action_history.append(tf.cast(action_numbers, dtype = tf.int32))
			state_history.append(state)
			state_next_history.append(next_state)
			reward_history.append(reward)
			map_history.append(map)
			
			# -- ADVANCE STATE --
			recreation = next_recreation
			state = next_state
			
			if step % UPDATE_AFTER_ACTIONS == 0 and len(action_history) > BATCH_SIZE:
				#should do a random selection not just the last batch_size
				state_next_sampe				= state_next_history[-BATCH_SIZE:]  
				reward_sample					= reward_history[-BATCH_SIZE:] 
				action_sample 					= action_history[-BATCH_SIZE:] 
				state_sample 					= state_history[-BATCH_SIZE:] 
				
				# -- PREDICT FUTURE REWARDS AND Q_VALUES USING STATE_NEXT --
				predicted_future_rewards 		= target_model.predict(tf.convert_to_tensor(state_next_sampe))
				predicted_future_reward_values	= tf.math.reduce_max(predicted_future_rewards, axis = -1)	#the action with the most reward is the one we would take
				#predicted_future_reward_values	= tf.expand_dims(predicted_future_reward_values, axis = -1)	#since we collapsed the last axis
				
				predicted_q_values				= tf.convert_to_tensor(reward_sample) + GAMMA * predicted_future_reward_values
				
				# -- CALCULATE Q_VALUES USING MODEL AND STATE_SAMPLE --
				masks							= tf.one_hot(action_sample, NUM_ACTIONS)
				with tf.GradientTape() as tape:
					q_values 		= model(tf.convert_to_tensor(state_sample))
					q_actions 		= tf.math.multiply(q_values, masks)	#apply mask to get q values of only the would-be-selected actions
					q_action_values = tf.reduce_sum(q_actions, axis = -1)
					#q_action_values	= tf.expand_dims(q_action_values, axis = -1)
					loss = LOSS_FUNCTION(predicted_q_values, q_action_values)
					
				# -- BACKPROPAGATION --
				gradients = tape.gradient(loss, model.trainable_variables)
				optimizer.apply_gradients(zip(gradients, model.trainable_variables))
				n_model_backprops += 1
				
			if step % UPDATE_TARGET_NETWORK == 0:
				final_reward_reducxed = reward_reduced
				target_model.set_weights(model.get_weights())
				n_target_updates += 1
				print("Updating target netowork.  %i step, %i model backpropagations, %i target model updates, %i random actions, %i model actions" % (step, n_model_backprops, n_target_updates, n_rand_actions, n_model_actions))
				print(reward_reduced)	

		save_state(best_state, image_ndx, epoch_iter, fount_at_step, best_reward)
		best_recreation_values[image_ndx] = best_reward / (TARGET_IMG_SHAPE[0] * TARGET_IMG_SHAPE[1])
		print(best_reward / (TARGET_IMG_SHAPE[0] * TARGET_IMG_SHAPE[1]))
		print(np.mean(best_recreation_values))
		print(np.std(best_recreation_values))
				
		
		'''
		axes[0][0].imshow(left_ref)
		axes[0][1].imshow(right_src)
		axes[1][0].imshow(recreation, cmap = "gray")
		axes[1][1].imshow(disparity_x_for_drawing(map), cmap = "gray")
		plt.show()
		print(reward_reduced)
		_ = input("so what happened here?")
		'''	
			
			
		
		#gc will probably take care of this but just-in-case
		del action_history
		del state_history
		del state_next_history
		del reward_history
		
		
	epoch_iter += 1																		# the number of times we will go through the whole 

	
	print("final values")
	print(np.mean(best_recreation_values))
	print(np.std(best_recreation_values))
	
_ = input("we are done, observe and despair")

	



