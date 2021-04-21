import tensorflow as tf

def update_map_x(map, action_numbers, map_shape):	
	change = tf.subtract(action_numbers, 1)
	change = tf.expand_dims(change, axis = -1)
	
	change = tf.cast(change, dtype = tf.int32)

	new_map = tf.math.add(map, change)
	new_map = tf.clip_by_value(new_map, clip_value_min = (-1 * map_shape[1]), clip_value_max = 0)
	new_map = tf.cast(new_map, dtype = tf.int32)
	
	return new_map
	
def update_map_x_grey(map, action_numbers, map_shape):	
	return update_map_x(map, action_numbers, map_shape)