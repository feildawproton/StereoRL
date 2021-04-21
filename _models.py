import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(state_shape, kernel_dims, num_actions, filters_per_layer):
	initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 1.0)
	
	input	= layers.Input(shape = state_shape)
	
	layer_1 = layers.Conv2D(filters = filters_per_layer, kernel_size = kernel_dims, padding = "same", activation="relu", kernel_initializer = initializer)(input)
	norm_1 	= layers.BatchNormalization()(layer_1, training = True)
	layer_2 = layers.Conv2D(filters = filters_per_layer, kernel_size = kernel_dims, padding = "same", activation="relu", kernel_initializer = initializer)(norm_1)
	norm_2 	= layers.BatchNormalization()(layer_2, training = True)
	layer_3 = layers.Conv2D(filters = filters_per_layer, kernel_size = kernel_dims, padding = "same", activation="relu", kernel_initializer = initializer)(norm_2)
	norm_3 	= layers.BatchNormalization()(layer_3, training = True)
	layer_4 = layers.Conv2D(filters = filters_per_layer, kernel_size = kernel_dims, padding = "same", activation="relu", kernel_initializer = initializer)(norm_3)
	norm_4	= layers.BatchNormalization()(layer_4, training = True)
	
	#relu because outputs are relative probabilities and predicted future rewards
	actions = layers.Conv2D(filters = num_actions, kernel_size = (3,3), padding = "same", activation = "relu")(norm_4)
	
	return keras.Model(inputs = input, outputs=actions)
	
def create_temporal_model(state_shape, kernel_dims, num_actions, filters_per_layer):
	initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 1.0)
	
	input	= layers.Input(shape = state_shape)
	
	layer_1 = layers.Conv2D(filters = filters_per_layer, kernel_size = kernel_dims, padding = "same", activation="relu", kernel_initializer = initializer)(input)
	#layer_2 = layers.Conv2D(filters = filters_per_layer, kernel_size = kernel_dims, padding = "same", activation="relu", kernel_initializer = initializer)(layer_1)
	#layer_3 = layers.Conv2D(filters = filters_per_layer, kernel_size = kernel_dims, padding = "same", activation="relu", kernel_initializer = initializer)(layer_2)
	
	convlstm_layer = tf.keras.layers.ConvLSTM2D(filters = filters_per_layer, kernel_size = kernel_dims, padding = "same")(layer_1)
	
	#relu because outputs are relative probabilities and predicted future rewards
	actions = layers.Conv2D(filters = num_actions, kernel_size = (3,3), padding = "same", activation = "relu")(convlstm_layer)
	
	return keras.Model(inputs = input, outputs=actions)
	