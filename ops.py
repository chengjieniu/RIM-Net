import numpy as np 
import tensorflow as tf

def num_params():
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

def init_weights(shape, name):
	return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def init_biases(shape):
	return tf.Variable(tf.zeros(shape))

def lrelu(x, leak=0.02):
	return tf.maximum(x, leak*x)

def batch_norm(input, phase_train):
	#return tf.contrib.layers.batch_norm(input, decay=0.999, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_train)
	return tf.contrib.layers.instance_norm(input)

def linear(input_, output_size, scope, add_reg=False):
	shape = input_.get_shape().as_list()
	with tf.variable_scope(scope):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], initializer=tf.random_normal_initializer(stddev=0.02))
		bias = tf.get_variable("bias", [output_size], initializer=tf.zeros_initializer())
		if add_reg:
			tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, matrix)
		print("linear","in",shape,"out",(shape[0],output_size))
		return tf.matmul(input_, matrix) + bias

def conv1d(input_, output_size, kernel_size, name, padding="SAME", add_reg=False):
	shape = input_.get_shape().as_list()
	with tf.variable_scope(name):
		matrix = tf.get_variable('Matrix', [1, shape[2], output_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
		bias = tf.get_variable('bias', [output_size], initializer=tf.zeros_initializer())
		if add_reg:
			tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, matrix)
		conv = tf.nn.conv1d(input_, matrix, stride=kernel_size, padding=padding)
		conv = tf.nn.bias_add(conv, bias)
		print("conv1d","in",input_.shape,"out",conv.shape)
		return conv

def mlp_conv(input_, layer_dim):
	shape = input_.get_shape().as_list()
	for i, num_outputs in enumerate(layer_dim[:-1]):
		input_ = tf.layers.conv1d(
			input_, num_outputs, kernel_size=1, name='conv_%d' % i
		)
	outputs = tf.layers.conv1d(
		input_, layer_dim[-1], kernel_size=1, name='conv_%d' % (len(layer_dim)-1)
	)
	shape_out = outputs.get_shape().as_list()
	print("mlp_conv", "in", shape, "out", shape_out )
	return outputs

def mlp_conv_topnet(input_, layer_dim, level,j):
	shape = input_.get_shape().as_list()
	for i, num_outputs in enumerate(layer_dim[:-1]):
		input_ = conv1d(
			input_, num_outputs, kernel_size=1, name='conv_%d' % i
		)
		input_ = lrelu(input_)
	outputs = conv1d(
		input_, layer_dim[-1], kernel_size=1, name='conv_%d_%d' % (level, j)
	)
	shape_out = outputs.get_shape().as_list()
	print("mlp_conv", "in", shape, "out", shape_out )
	return outputs

def conv2d(input_, shape, strides, scope, padding="SAME", add_reg=False):
	with tf.variable_scope(scope):
		matrix = tf.get_variable('Matrix', shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
		bias = tf.get_variable('bias', [shape[-1]], initializer=tf.zeros_initializer())
		if add_reg:
			tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, matrix)
		conv = tf.nn.conv2d(input_, matrix, strides=strides, padding=padding)
		conv = tf.nn.bias_add(conv, bias)
		print("conv2d","in",input_.shape,"out",conv.shape)
		return conv

def conv2d_nobias(input_, shape, strides, scope, padding="SAME", add_reg=False):
	with tf.variable_scope(scope):
		matrix = tf.get_variable('Matrix', shape, initializer=tf.contrib.layers.xavier_initializer())#tf.truncated_normal_initializer(stddev=0.02)
		if add_reg:
			tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, matrix)
		conv = tf.nn.conv2d(input_, matrix, strides=strides, padding=padding)
		print("conv2d","in",input_.shape,"out",conv.shape)
		return conv

def conv3d(input_, shape, strides, scope, padding="SAME"):
	with tf.variable_scope(scope):
		matrix = tf.get_variable("Matrix", shape, initializer=tf.contrib.layers.xavier_initializer())
		bias = tf.get_variable("bias", [shape[-1]], initializer=tf.zeros_initializer())
		conv = tf.nn.conv3d(input_, matrix, strides=strides, padding=padding)
		conv = tf.nn.bias_add(conv, bias)
		print("conv3d","in",input_.shape,"out",conv.shape)
		return conv

def deconv3d(input_, shape, out_shape, strides, scope, padding="SAME"):
	with tf.variable_scope(scope):
		matrix = tf.get_variable("Matrix", shape, initializer=tf.contrib.layers.xavier_initializer())
		bias = tf.get_variable("bias", [shape[-2]], initializer=tf.zeros_initializer())
		conv = tf.nn.conv3d_transpose(input_, matrix, out_shape, strides=strides, padding=padding)
		conv = tf.nn.bias_add(conv, bias)
		print("deconv3d","in",input_.shape,"out",conv.shape)
		return conv
 
