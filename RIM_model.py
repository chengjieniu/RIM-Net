import os
import math
from glob import glob
import numpy as np
import tensorflow as tf
import h5py
import time

from ops import *

import mcubes

class RIM(object):
	def __init__(self, sess, real_size, points_per_shape, is_training = False, z_dim=128, ef_dim=32, gf_dim=256, dataset_name='default', checkpoint_dir=None, result_dir=None, data_dir='./data', branch_num=None):

		self.sess = sess

		self.real_size = real_size #output point-value voxel grid size in training
		self.points_per_shape = points_per_shape #training batch size (virtual, batch_size is the real batch_size)
		
		self.batch_size = self.points_per_shape
		
		self.input_size = 64 #input voxel grid size

		self.z_dim = z_dim
		self.ef_dim = ef_dim
		self.gf_dim = gf_dim

		self.dataset_name = dataset_name
		self.checkpoint_dir = checkpoint_dir
		self.data_dir = data_dir

		self.gf_split = branch_num
		self.nlevels = int(math.log2(self.gf_split))

		if is_training:
			data_hdf5_name = self.data_dir+'/'+self.dataset_name+'.hdf5'
		else:
			data_hdf5_name = self.data_dir+'/'+self.dataset_name.replace("train", "test") +'.hdf5'
		
		if os.path.exists(data_hdf5_name):
			self.data_dict = h5py.File(data_hdf5_name, 'r')
			data_points_int = self.data_dict['points_'+str(self.real_size)][:]
			self.data_points = (data_points_int+0.5)/self.real_size-0.5
			self.data_values = self.data_dict['values_'+str(self.real_size)][:]
			self.data_voxels = self.data_dict['voxels'][:]
			if self.points_per_shape!=self.data_points.shape[1]:
				print("error: points_per_shape!=data_points.shape")
				exit(0)
			if self.input_size!=self.data_voxels.shape[1]:
				print("error: input_size!=data_voxels.shape")
				exit(0)
		else:
			print("error: cannot load "+data_hdf5_name)
			exit(0)
	
		if True:
			self.real_size = 64 #output point-value voxel grid size in testing
			self.test_size = 32 #related to testing batch_size, adjust according to gpu memory size
			self.batch_size = self.test_size*self.test_size*self.test_size #do not change
			
			#get coords
			dima = self.test_size
			dim = self.real_size
			self.aux_x = np.zeros([dima,dima,dima],np.uint8)
			self.aux_y = np.zeros([dima,dima,dima],np.uint8)
			self.aux_z = np.zeros([dima,dima,dima],np.uint8)
			multiplier = int(dim/dima)
			multiplier2 = multiplier*multiplier
			multiplier3 = multiplier*multiplier*multiplier
			for i in range(dima):
				for j in range(dima):
					for k in range(dima):
						self.aux_x[i,j,k] = i*multiplier
						self.aux_y[i,j,k] = j*multiplier
						self.aux_z[i,j,k] = k*multiplier
			self.coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						self.coords[i*multiplier2+j*multiplier+k,:,:,:,0] = self.aux_x+i
						self.coords[i*multiplier2+j*multiplier+k,:,:,:,1] = self.aux_y+j
						self.coords[i*multiplier2+j*multiplier+k,:,:,:,2] = self.aux_z+k
			self.coords = (self.coords+0.5)/dim-0.5
			self.coords = np.reshape(self.coords,[multiplier3,self.batch_size,3])

		self.build_model()

	def build_model(self):
		
		# gout[0]:[batchsize, 2] gout[1]:[batchsize, 4] gout[2]:[batchsize, 8] gout[3]:[batchsize, 16]
		# levelmax[0]:[batchsize, 1] levelmax[1]:[batchsize, 1] levelmax[2]:[batchsize, 1] levelmax[3]:[batchsize, 1]
		# levelparent[0]:[batchsize, 1] levelparent[1]:[batchsize, 2] levelparent[2]:[batchsize, 4] levelparent[3]:[batchsize, 8]
		self.vox3d = tf.placeholder(shape=[1,self.input_size,self.input_size,self.input_size,1], dtype=tf.float32, name="vox3d")
		self.z_vector = tf.placeholder(shape=[1,self.z_dim], dtype=tf.float32, name="z_vector")
		self.point_coord = tf.placeholder(shape=[None,3], dtype=tf.float32, name="point_coord")
		self.point_value = tf.placeholder(shape=[None,1], dtype=tf.float32, name="point_value")
		
		self.E = self.encoder(self.vox3d, phase_train=True, reuse=False)
		self.gout, self.levelmax_, self.levelparent_ = self.generator(self.point_coord, self.E, phase_train=True, reuse=False)
		
		self.sE = self.encoder(self.vox3d, phase_train=False, reuse=True)
		self.sgout, self.slevelmax_, self.slevelparent_= self.generator(self.point_coord, self.sE, phase_train=False, reuse=True)
		self.Bgout, self.Blevelmax_, self.Blevelparent_ = self.generator(self.point_coord, self.z_vector, phase_train=False, reuse=True)

		self.lossrecon = []
		
		# reconstruction loss
		for i in range(self.nlevels):
			self.lossrecon.append(tf.reduce_mean(tf.square(self.point_value - self.levelmax_[i])))


		self.losscover = []

		# decomposition loss
		for i in range(self.nlevels-1):
			self.losscover.append(tf.reduce_mean(tf.square(self.gout[i] - self.levelparent_[i+1])))
		
		self.losscomb = []

		for i in range(1,self.nlevels):
			self.losscomb.append(self.lossrecon[i] + 10 * self.losscover[i-1])
		self.loss = tf.reduce_mean(self.lossrecon) 
	
		self.saver = tf.train.Saver(max_to_keep=1000)

	# Feature Decoder
	def create_split(self, level, input_channels, output_channels, inputs, reuse=False):
		with tf.variable_scope("slevel_%d" % (level)) as scope:
			if reuse:
				scope.reuse_variables()
			slevel = lrelu(linear(inputs, input_channels , 'split_1'))
			slevel = tf.nn.sigmoid(linear(slevel, output_channels*2, 'split_2')) 
			slevel = tf.reshape(slevel, [tf.shape(inputs)[0], -1, output_channels])
		return slevel
	
	# Part Decoder
	def create_branch(self, level, input_channels, output_channels, inputs, reuse=False):
		with tf.variable_scope("blevel_%d" % (level)) as scope:
			if reuse:
				scope.reuse_variables()
			blevel = lrelu(linear(inputs, input_channels*4, 'branch_l1'))
			blevel = lrelu(linear(blevel, input_channels, 'branch_l2'))
			blevel = linear(blevel, output_channels*2, 'branch_l3')
			blevel = tf.reshape(blevel, [tf.shape(inputs)[0], -1, output_channels])
		return blevel

	# Per-point Gaussians as local point distributions
	def gaussian(self, inputs, points, level):

		pointsc = tf.expand_dims(points, dim=1)
		pointsc = tf.tile(pointsc, [1,2**level,1])
		px = pointsc[:,:,0]
		py = pointsc[:,:,1]
		pz = pointsc[:,:,2]

		c = tf.abs(inputs[:,0,:])
		c = tf.clip_by_value(c,1e-6,1)

		x_mean = inputs[:,1,:]  #[1024,5,100] 
		y_mean = inputs[:,2,:] #[1024,5,100]
		z_mean = inputs[:,3,:]  #[1024,5,100]
		r_x =  tf.sigmoid(inputs[:,4,:]) #[1024,5,100]
		r_x =  tf.maximum(1e-6, r_x)
		r_y =  tf.sigmoid(inputs[:,5,:]) #[1024,5,100]
		r_y =  tf.maximum(1e-6, r_y)
		r_z =  tf.sigmoid(inputs[:,6,:]) #[1024,5,100]
		r_z =  tf.maximum(1e-6, r_z)

		f = c * tf.exp(-((x_mean-px)**2/(2*(r_x**2)) + (y_mean-py)**2/(2*(r_y**2)) + (z_mean-pz)**2/(2*(r_z**2))))
		return f
	
	def generator(self, points, z, phase_train=True, reuse=False):
		
		batch_size = tf.shape(points)[0]
		zs = tf.tile(z, [batch_size,1])
		pointz = tf.concat([points,zs],1)
		print("pointz",pointz.shape)

		nlevels = int(math.log2(self.gf_split))		

		Nins = self.gf_dim
		Ninb = self.gf_dim
		Nout = self.z_dim
		para_num = 7
		
		#level 1
		level0 = self.create_split(0,Nins, Nout, zs, reuse )
		outs = [level0, ]

		# feature decoder process
		for i in range(1, nlevels-1):
			temp = []				
			inp = outs[-1]
			for j in range(2**i):
				if j== 0:
					temp=self.create_split(i, Nins, Nout, inp[:,j,:], reuse)
				else:
					temp=tf.concat([temp,self.create_split(i, Nins, Nout, inp[:,j,:], True)],1)

			outs.append(tf.reshape(temp, [tf.shape(points)[0], -1, Nout]))
			
		# part decoder process
		blevel0 = self.create_branch(0, Ninb, para_num, pointz, reuse )
		bouts = [blevel0, ]

		for i in range(1, nlevels):
			temp = []
			inp = outs[i-1]
			binp = tf.expand_dims(points, 1)
			binp = tf.tile(binp, [1, tf.shape(inp)[1], 1])
			binp = tf.concat([binp, inp], 2)	
			for j in range(2**i):
				if j== 0:
					temp=self.create_branch(i, Nins, para_num, binp[:,j,:], reuse)
				else:
					temp=tf.concat([temp,self.create_branch(i, Nins, para_num, binp[:,j,:], True)],1)
			bouts.append(tf.reshape(temp, [tf.shape(points)[0], -1, para_num]))

		#per-point Gaussian process
		gout = []
		for j in range(nlevels):
			ginp = tf.transpose(bouts[j], [0,2,1])
			gout.append(
				self.gaussian(ginp, points, (1+j))
			)
			# gout[0]:[batchsize, 2]
			# gout[1]:[batchsize, 4]
			# gout[2]:[batchsize, 8]
			# gout[3]:[batchsize, 16]
			
			# level max
		levelmax = []
		levelparent = []
		for j in range(nlevels):
			temp = []
			levelmax.append(tf.reduce_max(gout[j], axis=1, keepdims=True))
			for z in range(2**j):
				if z==0:
					temp=tf.reduce_max(gout[j][:,2*z:2*(z+1)], axis=1, keepdims=True)
				else:
					temp=tf.concat([temp, tf.reduce_max(gout[j][:,2*z:2*(z+1)], axis=1, keepdims=True)], 1)
			levelparent.append(tf.reshape(temp, [tf.shape(points)[0], -1]))

			# levelmax[0]:[batchsize, 1] levelmax[1]:[batchsize, 1] levelmax[2]:[batchsize, 1] levelmax[3]:[batchsize, 1]
			# levelparent[0]:[batchsize, 1] levelparent[1]:[batchsize, 2] levelparent[2]:[batchsize, 4] levelparent[3]:[batchsize, 8]
		
		print("the network is done")			
		

		return gout, levelmax, levelparent
	
	# 3D encoder
	def encoder(self, inputs, phase_train=True, reuse=False):
		with tf.variable_scope("encoder") as scope:
			if reuse:
				scope.reuse_variables()
			
			d_1 = conv3d(inputs, shape=[4, 4, 4, 1, self.ef_dim], strides=[1,2,2,2,1], scope='conv_1')
			d_1 = lrelu(batch_norm(d_1, phase_train))

			d_2 = conv3d(d_1, shape=[4, 4, 4, self.ef_dim, self.ef_dim*2], strides=[1,2,2,2,1], scope='conv_2')
			d_2 = lrelu(batch_norm(d_2, phase_train))
			
			d_3 = conv3d(d_2, shape=[4, 4, 4, self.ef_dim*2, self.ef_dim*4], strides=[1,2,2,2,1], scope='conv_3')
			d_3 = lrelu(batch_norm(d_3, phase_train))

			d_4 = conv3d(d_3, shape=[4, 4, 4, self.ef_dim*4, self.ef_dim*8], strides=[1,2,2,2,1], scope='conv_4')
			d_4 = lrelu(batch_norm(d_4, phase_train))

			d_5 = conv3d(d_4, shape=[4, 4, 4, self.ef_dim*8, self.z_dim], strides=[1,1,1,1,1], scope='conv_5', padding="VALID")
			d_5 = tf.nn.sigmoid(d_5)
		
			return tf.reshape(d_5,[1,self.z_dim])
	
	def train(self, config):

		# progressive training
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		supdate_list = []
		bupdate_list = []
		encoder_list = []
		encoder_list.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder" ))
		encoder_list.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="blevel_0" ))
		for i in range(1, self.nlevels):	
			supdate_list.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="slevel_%d" % (i-1) ))
			bupdate_list.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="blevel_%d" % (i) ))

		with tf.control_dependencies(update_ops):
			ae_optim = []
			for i in range(self.nlevels):
				if i==0:
					ae_optim.append(tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).\
						minimize(self.lossrecon[0], var_list=[encoder_list[0], encoder_list[1]]))
				else:
					ae_optim.append(tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).\
						minimize(self.losscomb[i-1], var_list= [supdate_list[i-1], bupdate_list[i-1]]))
			ae_optim.append(tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss))

		self.sess.run(tf.global_variables_initializer())
		
		batch_idxs = len(self.data_points)
		batch_index_list = np.arange(batch_idxs)
		
		print("\n\n----------net summary----------")
		print("training samples   ", batch_idxs)
		print("network branch	 ", self.gf_split)
		print("-------------------------------\n\n")
		
		counter = 0 
		start_time = time.time()

		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			counter = checkpoint_counter+1
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")	
		
		# -------- training --------
		assert config.epoch==0 or config.iteration==0
		training_epoch = config.epoch + int(config.iteration/batch_idxs)

		addnum = int(counter/training_epoch) + 1
		while counter<training_epoch*(1+self.nlevels):
			for epoch in range(counter, training_epoch*addnum+1):
				# unsupervised training
				np.random.shuffle(batch_index_list)
				avg_loss = 0
				avg_num = 0

				rl=np.array([0.0,0.0,0.0,0.0])
				cl=np.array([0.0,0.0,0.0])

				for idx in range(batch_idxs):
					dxb = batch_index_list[idx]					
					_, errAE, gout, lossrecon, losscover = self.sess.run([ae_optim[addnum-1], self.loss, self.gout, self.lossrecon, self.losscover],
							feed_dict={
								self.vox3d: self.data_voxels[dxb:dxb+1],
								self.point_coord: self.data_points[dxb],
								self.point_value: self.data_values[dxb],
							})
					
					avg_loss += errAE
					avg_num += 1 

					# recon loss
					for i in range(self.nlevels):
						rl[i] = rl[i]+lossrecon[i]

					# decomp loss
					for i in range(self.nlevels-1):
						cl[i] = cl[i]+losscover[i]*10

					if addnum-1==0:
						print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f, currentloss: %.6f" %\
						(epoch, training_epoch*addnum, idx, batch_idxs, time.time() - start_time, avg_loss/avg_num, rl[addnum-1]/avg_num))
					elif addnum-1>0 and addnum-1<self.nlevels:
						print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f, currentloss: %.6f" %\
						(epoch, training_epoch*addnum, idx, batch_idxs, time.time() - start_time, avg_loss/avg_num, cl[addnum-2]/avg_num))
					else:
						print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f" %\
						(epoch, training_epoch*addnum, idx, batch_idxs, time.time() - start_time, avg_loss/avg_num))
				
				if epoch%int(5)==0:
					self.save(config.checkpoint_dir, epoch)
					self.test_1(config, "train1_"+str(self.real_size)+"_"+str(epoch))

			if training_epoch%int(5)!=0:
				self.save(config.checkpoint_dir, training_epoch*addnum)
			
			counter = counter + training_epoch +1
			addnum = addnum+1

	
	# -------- quantitative evaluation --------
	def test_1(self, config, name):
		if not os.path.exists(config.result_dir.replace("test", "temp")):
			os.makedirs(config.result_dir.replace("test", "temp"))
		color_list = ["255 0 0","0 255 0","0 0 255","255 255 0","255 0 255","0 255 255","180 180 180", "100 100 100", \
			"255 128 128","128 255 128","128 128 255","255 255 128","255 128 255","128 255 255", "30 144 255", "135 206 250"]
		multiplier = int(self.real_size/self.test_size)
		multiplier2 = multiplier*multiplier
		
		# t = np.random.randint(len(self.data_voxels))
		for t in range(1,5):
			model_float = dict()
			for x in range(1, self.nlevels+1):
				model_float[x] = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2, 2**x],np.float32)

			batch_voxels = self.data_voxels[t:t+1]

			z_out = self.sess.run(self.sE,
				feed_dict={
					self.vox3d: batch_voxels,
				})
			
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						gout = self.sess.run(self.Bgout,
								feed_dict={
									self.z_vector: z_out,
									self.point_coord: self.coords[minib],
								})
						for x in range(1, self.nlevels + 1):
							model_float[x][self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1,:] = np.reshape(gout[x-1], [self.test_size,self.test_size,self.test_size, 2**x])

			thres = 0.4
			for x in range(1, self.nlevels+1):	
				vertices_num = 0
				triangles_num = 0
				vertices_list = []
				triangles_list = []
				vertices_num_list = [0]
				for split in range(2**x):
					vertices, triangles = mcubes.marching_cubes(model_float[x][:,:,:,split], thres)
					vertices_num += len(vertices)
					triangles_num += len(triangles)
					vertices_list.append(vertices)
					triangles_list.append(triangles)
					vertices_num_list.append(vertices_num)
					
				#output ply
				fout = open(config.result_dir.replace("test", "temp")+"/"+name+"_"+str(t)+"_vox_%d.ply"%(x), 'w')
				fout.write("ply\n")
				fout.write("format ascii 1.0\n")
				fout.write("element vertex "+str(vertices_num)+"\n")
				fout.write("property float x\n")
				fout.write("property float y\n")
				fout.write("property float z\n")
				fout.write("property uchar red\n")
				fout.write("property uchar green\n")
				fout.write("property uchar blue\n")
				fout.write("element face "+str(triangles_num)+"\n")
				fout.write("property uchar red\n")
				fout.write("property uchar green\n")
				fout.write("property uchar blue\n")
				fout.write("property list uchar int vertex_index\n")
				fout.write("end_header\n")
					
				for split in range(2**x):
					vertices = (vertices_list[split])/self.real_size-0.5
					for i in range(len(vertices)):
						color = color_list[split]
						fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
					
				for split in range(2**x):
					triangles = triangles_list[split] + vertices_num_list[split]
					for i in range(len(triangles)):
						color = color_list[split]
						fout.write(color+" 3 "+str(triangles[i,0])+" "+str(triangles[i,1])+" "+str(triangles[i,2])+"\n")
				
				fout.close()
				print("[sample]"+str(t))

	def test_dae(self, config):

		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		color_list = ["255 0 0","0 255 0","0 0 255","255 255 0","255 0 255","0 255 255","255 128 128", "100 100 100", \
			"180 180 180","128 255 128","128 128 255","255 255 128","255 128 255","128 255 255", "30 144 255", "135 206 250"]
		
		self.real_size = 64
		dima = self.test_size
		dim = self.real_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		batch_idxs = len(self.data_points)
		for t in range(0,batch_idxs):
			model_float = dict()
			for x in range(1, self.nlevels+1):
				model_float[x] = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2, 2**x],np.float32)
			batch_voxels = self.data_voxels[t:t+1]
			z_out = self.sess.run(self.sE,
				feed_dict={
					self.vox3d: batch_voxels,
				})
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						gout = self.sess.run(self.Bgout,
							feed_dict={
								self.z_vector: z_out,
								self.point_coord: self.coords[minib],
							})
						for x in range(1, self.nlevels + 1):
							model_float[x][self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1,:] = np.reshape(gout[x-1], [self.test_size,self.test_size,self.test_size, 2**x])

			thres = 0.4
			for x in range(1, self.nlevels+1):	
				vertices_num = 0
				triangles_num = 0
				vertices_list = []
				triangles_list = []
				vertices_num_list = [0]
				for split in range(2**x):
					vertices, triangles = mcubes.marching_cubes(model_float[x][:,:,:,split], thres)
					vertices_num += len(vertices)
					triangles_num += len(triangles)
					vertices_list.append(vertices)
					triangles_list.append(triangles)
					vertices_num_list.append(vertices_num)
				
				#output ply
				fout = open(config.result_dir+"/"+str(t)+"_avox_%d.ply"%(x), 'w')
				fout.write("ply\n")
				fout.write("format ascii 1.0\n")
				fout.write("element vertex "+str(vertices_num)+"\n")
				fout.write("property float x\n")
				fout.write("property float y\n")
				fout.write("property float z\n")
				fout.write("property uchar red\n")
				fout.write("property uchar green\n")
				fout.write("property uchar blue\n")
				fout.write("element face "+str(triangles_num)+"\n")
				fout.write("property uchar red\n")
				fout.write("property uchar green\n")
				fout.write("property uchar blue\n")
				fout.write("property list uchar int vertex_index\n")
				fout.write("end_header\n")
					
				for split in range(2**x):
					vertices = (vertices_list[split])/self.real_size-0.5
					for i in range(len(vertices)):
						color = color_list[split]
						fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
					
				for split in range(2**x):
					triangles = triangles_list[split] + vertices_num_list[split]
					for i in range(len(triangles)):
						color = color_list[split]
						fout.write(color+" 3 "+str(triangles[i,0])+" "+str(triangles[i,1])+" "+str(triangles[i,2])+"\n")
				
				#output separated files for different parts
				if t!=-1:
					vertices, triangles = mcubes.marching_cubes(batch_voxels[0,:,:,:,0], thres)
					#output input vox ply
					fout1 = open(config.result_dir+"/"+str(t)+"_input.ply", 'w')
					fout1.write("ply\n")
					fout1.write("format ascii 1.0\n")
					fout1.write("element vertex "+str(len(vertices))+"\n")
					fout1.write("property float x\n")
					fout1.write("property float y\n")
					fout1.write("property float z\n")
					fout1.write("property uchar red\n")
					fout1.write("property uchar green\n")
					fout1.write("property uchar blue\n")
					fout1.write("element face "+str(len(triangles))+"\n")
					fout1.write("property uchar red\n")
					fout1.write("property uchar green\n")
					fout1.write("property uchar blue\n")
					fout1.write("property list uchar int vertex_index\n")
					fout1.write("end_header\n")
					color = "180 180 180"
					vertices = (vertices)/self.real_size-0.5
					for i in range(len(vertices)):
						fout1.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
					for i in range(len(triangles)):
						fout1.write(color+" 3 "+str(triangles[i,0])+" "+str(triangles[i,1])+" "+str(triangles[i,2])+"\n")
					fout1.close()
					
					for split in range(2**x):
						vertices = (vertices_list[split])/self.real_size-0.5
						triangles = triangles_list[split]
						#output part ply
						fout1 = open(config.result_dir+"/"+str(t)+"_vox_%d_"%(x)+str(split)+".ply", 'w')
						fout1.write("ply\n")
						fout1.write("format ascii 1.0\n")
						fout1.write("element vertex "+str(len(vertices))+"\n")
						fout1.write("property float x\n")
						fout1.write("property float y\n")
						fout1.write("property float z\n")
						fout1.write("property uchar red\n")
						fout1.write("property uchar green\n")
						fout1.write("property uchar blue\n")
						fout1.write("element face "+str(len(triangles))+"\n")
						fout1.write("property uchar red\n")
						fout1.write("property uchar green\n")
						fout1.write("property uchar blue\n")
						fout1.write("property list uchar int vertex_index\n")
						fout1.write("end_header\n")
						for i in range(len(vertices)):
							color = color_list[split]
							fout1.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
						for i in range(len(triangles)):
							color = color_list[split]
							fout1.write(color+" 3 "+str(triangles[i,0])+" "+str(triangles[i,1])+" "+str(triangles[i,2])+"\n")
						fout1.close()
				
				fout.close()
			
				print("[result]"+ str(t))


	@property
	def model_dir(self):
		return "{}".format(
				self.dataset_name[:8])
			
	def save(self, checkpoint_dir, step):
		model_name = "RIM.model"
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0
