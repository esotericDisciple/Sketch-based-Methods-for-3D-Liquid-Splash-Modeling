from __future__ import division
from tensorflow.python.client import device_lib
from random import shuffle

import numpy as np
import tensorflow as tf

import uniio
import argparse
import os
import json
import glob
import random
import collections
import math
import time

CONV3D_AE   = collections.namedtuple('CONV3D_AE', ('decoded', 'loss', 'loss_occ', 'loss_vel', 'optimizer', 'encoded'))
SCENES      = collections.namedtuple('SCENES', ('sceneNames', 'simIdStart', 'simIdEnd', 'frameIdStart', 'frameIdEnd'))

def parse_args(model_restore, model_mode):
	parser = argparse.ArgumentParser(description='Sketch-based Methods for 3D Liquid Splash Modeling')
	parser.add_argument('--mode', default='train', choices=['train', 'test'])
	parser.add_argument('--seed', type=int)

	parser.add_argument('--num_epoch', default=30, type=int, help='the number of epochs')
	parser.add_argument('--batch_size', default=4, type=int, help='mini batch size')
	parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate of optimizer')

	parser.add_argument('--train_dir', default='./data/train', type=str, help='path to training dataset')
	parser.add_argument('--manta_dir', default='../manta/sbfs_scenes/data', type=str, help='path to levelset data')
	parser.add_argument('--mat_dir', default='../VisMat/data', type=str, help='path to streamline data')
	
	parser.add_argument('--result_dir', default='./data_extend/result', type=str, help='path to outputs')
	parser.add_argument('--log_dir', default='./log_extend', type=str, help='path to logfile')
	parser.add_argument('--checkpoint_dir', default='./checkpoint_extend', type=str, help='path to checkpoint')
	args = parser.parse_args()

	make_dirs([args.result_dir, args.log_dir, args.checkpoint_dir])
	# reuse previously saved configuation
	if model_restore and os.path.isfile(os.path.join(args.result_dir, 'options.json')):       
		with open(os.path.join(args.result_dir, 'options.json')) as f:
			json_dict = json.load(f)
			argparse_dict = vars(args)
			argparse_dict.update(json_dict)
	# set model mode
	args.mode=model_mode 
	# intialize random state
	random_seed(args)
	with open(os.path.join(args.result_dir, "options.json"), "w") as f:
		f.write(json.dumps(vars(args), sort_keys=True, indent=4))   
	
	return args

def random_seed(args):
	# Initialize internal state of the random number generator.
	if args.seed is None:
		args.seed = random.randint(0, 2**31 - 1)
	tf.set_random_seed(args.seed)
	np.random.seed(args.seed)
	rdm = np.random.RandomState(13)
	random.seed(args.seed) 

def make_dirs(dirs):
	"""create dir if not exists yet"""
	for dir in dirs:
		if not os.path.exists(dir):
			os.makedirs(dir)

def load_data_addrs_npy(data_dir):
	# read addresses from the 'train' folder
	data_path = data_dir + '/*flipLevelSet.npy'
	addrs = glob.glob(data_path)

	# to shuffle data
	random.shuffle(addrs)

	# Divide the data into 80% train and 20% test
	train_addrs = addrs[0:int(0.8*len(addrs))]
	test_addrs  = addrs[int(0.8*len(addrs)):]

	print('num traing data = {:04d}'.format(len(train_addrs)))
	# for i in range(len(train_addrs)):
	#     print(train_addrs[i])
	
	print('num testing data = {:04d}'.format(len(test_addrs)))
	# print(*test_addrs, sep = "\n")

	return train_addrs, test_addrs

def load_data_addrs_uni(args):
	# scene configuration
	Scene = SCENE(sceneName='sbfs_flip_obstacle_wineglass', 
					simIdStart=10000, simIdEnd= 10000 + 100, 
					frameIdStart= 20, frameIdEnd=100)

	# data address(filenames)
	addrs = []
	data_path_base = os.path.join(args.manta_dir, Scene.sceneName+'_%04d')
	for simNo in range(Scene.simIdStart, Scene.simIdEnd):
		data_path = data_path_base % simNo
		for frame in range(Scene.frameIdStart,Scene.frameIdEnd):
			infilepath = os.path.join(data_path, 'flipLevelSet_{:04d}.uni'.format(frame))
			addrs.append(infilepath)

	random.shuffle(addrs)
	train_addrs = addrs[0:int(0.8*len(addrs))]
	test_addrs  = addrs[int(0.8*len(addrs)):]

	print('num traing data = {:04d}'.format(len(train_addrs)))
	print('num testing data = {:04d}'.format(len(test_addrs)))
	# print(*test_addrs, sep = '\n')

	with open(os.path.join(args.result_dir, 'train_addrs.txt'), 'w') as f:
		for addr in train_addrs:
			f.write('%s\n' % addr)
	with open(os.path.join(args.result_dir, 'test_addrs.txt'), 'w') as f:
		for addr in test_addrs:
			f.write('%s\n' % addr)    

	return train_addrs, test_addrs   

def load_data_addrs_comp():
	filepath = './data/result'
	filename = 'test_addrs.txt'
	addrs = []
	with open(os.path.join(filepath, filename), 'r') as f:
		for line in f:
			(addr_x, addr_y)	= line.split()
			addr_lvst			= addr_y
			filename_lvst 		= addr_lvst[int(addr_lvst.rfind(os.sep)+1):int(addr_lvst.rfind('.'))]
			frame_num 			= filename_lvst[int(filename_lvst.rfind('_'))+1:]
			addr_lvst_pre 		= addr_lvst[0:int(addr_lvst.rfind(os.sep))]
			addr_lvst_pre_pre 	= addr_lvst_pre[0:int(addr_lvst_pre.rfind(os.sep))]
			simname	 			= addr_lvst_pre[int(addr_lvst_pre.rfind(os.sep)+1):]
			addr_z 				= addr_lvst_pre + os.sep + 'flipVel_' + frame_num + '.uni'
			addrs.append((addr_x, addr_y, addr_z))
	# print('num addrs_comp = {:04d}'.format(len(addrs)))
	print(addr_x, addr_y, addr_z)
	return addrs

def load_data_addrs(args, printfile):
	# scene configuration
	Scenes = SCENES(sceneNames=['sbfs_flip_water_pipes'],
					simIdStart=[10000], 	simIdEnd= [10499], 
					frameIdStart= [20], 	frameIdEnd=[99])

	# load sketch file and its corresponding levelset file
	addrs = []
	train_addrs = []
	test_addrs = []

	for sceneNo in range(len(Scenes.sceneNames)):	
		data_path_base_mat		= os.path.join(args.mat_dir, '4096', Scenes.sceneNames[sceneNo]+'_%04d')
		data_path_base_manta 	= os.path.join(args.manta_dir, Scenes.sceneNames[sceneNo]+'_%04d')
		for simNo in range(Scenes.simIdStart[sceneNo], Scenes.simIdEnd[sceneNo] + 1):
			data_path_mat 		= data_path_base_mat % simNo
			data_path_manta 	= data_path_base_manta % simNo
			for frameNo in range(Scenes.frameIdStart[sceneNo],Scenes.frameIdEnd[sceneNo] + 1):
				infilepath_sketch 	= os.path.join(data_path_mat, 'flipSketchGrid_{:04d}.bin'.format(frameNo))
				infilepath_lvst 	= os.path.join(data_path_manta, 'flipLevelSet_{:04d}.uni'.format(frameNo))
				infilepath_vel 		= os.path.join(data_path_manta, 'flipVel_{:04d}.uni'.format(frameNo))
				addrs.append((infilepath_sketch, infilepath_lvst, infilepath_vel))
				# check the file does exist
				if (not os.path.isfile(infilepath_sketch)) or ( not os.path.isfile(infilepath_lvst)) or ( not os.path.isfile(infilepath_vel)):
					print('{:}, {:} or {} does not exists!'.format(infilepath_sketch, infilepath_lvst, infilepath_vel))	
				if(frameNo % 10 == 0):
					test_addrs.append((infilepath_sketch, infilepath_lvst, infilepath_vel))
				elif (frameNo % 2 == 0):
					train_addrs.append((infilepath_sketch, infilepath_lvst, infilepath_vel))

	random.shuffle(train_addrs)
	random.shuffle(test_addrs)

	# random.shuffle(addrs)
	# train_addrs = addrs[0:int(0.8*len(addrs))]
	# test_addrs  = addrs[int(0.8*len(addrs)):]

	print('num traing data = {:04d}'.format(len(train_addrs)), flush=True, file=printfile)
	print('num testing data = {:04d}'.format(len(test_addrs)), flush=True, file=printfile)
	# print(*test_addrs, sep = '\n')

	# unpack feature and label
	with open(os.path.join(args.result_dir, 'train_addrs.txt'), 'w') as f:
		for (addr_x, addr_y, addr_z) in train_addrs:
			f.write('%s %s %s\n' % (addr_x, addr_y, addr_z))
	with open(os.path.join(args.result_dir, 'test_addrs.txt'), 'w') as f:
		for (addr_x, addr_y, addr_z) in test_addrs:
			f.write('%s %s %s\n' % (addr_x, addr_y, addr_z))

	return train_addrs, test_addrs

def get_next_batch_uni(data_addrs, ibatch, batch_size):
	data = []
	for i in range(ibatch*batch_size, ibatch*batch_size+batch_size):
		if os.path.isfile(data_addrs[i]):
			grid_header, grid_content = uniio.readUni(data_addrs[i])
			data.append(grid_content)
	data = np.asarray(data, dtype='float32')
	return data

def read_streamline_txt(filename):
	# load streamline data from txt file
	streamline_point_data 	= np.loadtxt(filename, dtype='float32')
	assert (streamline_point_data.shape[1] % 3 == 0) ,"Error in streamline file."
	num_streamline 			= streamline_point_data.shape[0]
	num_points 				= streamline_point_data.shape[1]//3

	# voxelize the streamline
	# warning: this step takes a long time, read raw grid data directly when possible
	# dimensions = [header['dimZ'], header['dimY'], header['dimX'], channels]
	grid_data = np.ndarray(shape=(128, 128, 128, 1), dtype='float32', order='C')
	for i in range(num_streamline):
		for j in range(num_points):
			p_x = int(round(streamline_point_data[i][3*j+0]))
			p_y = int(round(streamline_point_data[i][3*j+1]))
			p_z = int(round(streamline_point_data[i][3*j+2]))
			assert (p_x >= 0 and p_x <= 127) ,"Error in streamline starting point."
			assert (p_y >= 0 and p_y <= 127) ,"Error in streamline starting point."
			assert (p_z >= 0 and p_z <= 127) ,"Error in streamline starting point."
			# make sure the dimension order is aligned with levelset grid!
			# run check rountine in cpp!
			grid_data[p_z][p_y][p_x] = 1 

	return grid_data

def read_streamline_bin(filename):
	grid_data = None
	with open(filename, mode='rb') as file:
		grid_data = np.fromfile(file, dtype='float32')
		grid_data = np.reshape(grid_data, (128, 128, 128, 1),  order='C')
	return grid_data

def get_data_next_batch(data_addrs, ibatch, batch_size, export=False):
	data_streamline_occ = []
	data_streamline_vel = []
	data_grid_lvst 		= []
	data_grid_vel 		= []
	for i in range(ibatch*batch_size, ibatch*batch_size+batch_size):
		(data_sketch_addr, data_lvst_addr, data_vel_addr) = data_addrs[i]
		if os.path.isfile(data_sketch_addr) and os.path.isfile(data_lvst_addr) and os.path.isfile(data_vel_addr):
			grid_sketch_occ			= read_streamline_bin(data_sketch_addr)
			header_lvst, grid_lvst 	= uniio.readUni(data_lvst_addr)
			header_vel, grid_vel 	= uniio.readUni(data_vel_addr)

			# clean the velocity field, using levelset as mask(must be same as the streamline program!) 
			fluid_mask = (grid_lvst <= 1.0)
			grid_vel = grid_vel*fluid_mask

			# extract streamline velocity
			grid_sketch_vel = grid_sketch_occ*grid_vel

			# binarify levelset grid: should not include zero(ex. crop levelset grid is 0 everywhere)
			grid_lvst = np.array(grid_lvst < 0.0, dtype='float32')

			data_streamline_occ.append(grid_sketch_occ)
			data_streamline_vel.append(grid_sketch_vel)
			data_grid_lvst.append(grid_lvst)
			data_grid_vel.append(grid_vel)

			# export sketch and levelset file to check if the two grid data are aligned correctly
			# the visulization routine is in cpp file!
			if(export):
				uniio.writeUni('../VisFluid/data/sketch_occ_{:04}.uni'.format(i), header_lvst, grid_sketch_occ)
				uniio.writeUni('../VisFluid/data/sketch_vel_{:04}.uni'.format(i), header_vel, grid_sketch_vel)						
				uniio.writeUni('../VisFluid/data/levelset_{:04}.uni'.format(i), header_lvst, grid_lvst)
				uniio.writeUni('../VisFluid/data/vel_{:04}.uni'.format(i), header_vel, grid_vel)

		else:
			print('{:} or {:} does not exists!'.format(data_sketch_addr, data_lvst_addr))	

	data_streamline_occ = np.asarray(data_streamline_occ, dtype='float32')
	data_streamline_vel = np.asarray(data_streamline_vel, dtype='float32')
	data_grid_lvst 		= np.asarray(data_grid_lvst, dtype='float32')
	data_grid_vel 		= np.asarray(data_grid_vel, dtype='float32')

	# concatenate occupancy field and velocity field
	data_streamline = np.concatenate((data_streamline_occ, data_streamline_vel), axis=-1)
	data_grid 		= np.concatenate((data_grid_lvst, data_grid_vel), axis=-1)
	return data_streamline, data_grid

def gen_conv3d(batch_input, out_channels):
	 # [batch, in_depth, in_height, in_width, in_channels] => [batch, out_detpth, out_height, out_width, out_channels]
	 initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
	 return tf.layers.conv3d(batch_input, out_channels, kernel_size=3, strides=2, padding="same", kernel_initializer=initializer, name='conv')

def gen_deconv3d(batch_input, out_channels):
	# [batch, in_depth, in_height, in_width, in_channels] => [batch, out_detpth, out_height, out_width, out_channels]
	initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
	return tf.layers.conv3d_transpose(batch_input, out_channels, kernel_size=3, strides=2, padding="same", kernel_initializer=initializer, name='deconv')

def lrelu(x, alpha):
	with tf.name_scope("lrelu"):
		# adding these together creates the leak part and linear part
		# then cancels them out by subtracting/adding an absolute value term
		# leak: a*x/2 - a*abs(x)/2
		# linear: x/2 + abs(x)/2

		# this block looks like it has 2 inputs on the graph unless we do this
		# x = tf.identity(x)
		# return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)
		
		return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def batchnorm(inputs, training):
	initializer = tf.random_normal_initializer(mean=1.0, stddev=0.02)
	return tf.layers.batch_normalization(inputs, epsilon=1e-5, momentum=0.1, training=training, gamma_initializer=initializer)

def conv3d_enc_dec(data_x=None,	data_y=None, args=None, is_training=True, reuse=None):
	"""3D Convolutional Autoencoder network definition"""
	outchannels = 4

	layers = []

	layer_specs = [
		64,     # encoder_2: [batch, 64, 64, 64, 32] => [batch, 32, 32, 32, 64]
		128,    # encoder_3: [batch, 32, 32, 32, 64] => [batch, 16, 16, 16, 128]
		256,    # encoder_4: [batch, 16, 16, 16, 128] => [batch, 8, 8, 8, 256]
		256,    # encoder_5: [batch, 8, 8, 8, 256] => [batch, 4, 4, 4, 256]
		256,    # encoder_6: [batch, 4, 4, 4, 256] => [batch, 2, 2, 2, 256]
	]

	with tf.variable_scope('conv3d_autoencoder', reuse=reuse):
		# encoder_1: [batch, 128, 128, 128, inchannel] => [batch, 64, 64, 64, 32]
		with tf.variable_scope("encoder_1"):    
			output = gen_conv3d(data_x, 32)
			layers.append(output)

		for out_channels in layer_specs:
			with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
				rectified = lrelu(layers[-1], 0.2)  
				convolved = gen_conv3d(rectified, out_channels) # [batch, in_depth, in_height, in_width, in_channels] => [batch, in_depth/2, in_height/2, in_width/2, out_channels]
				output = batchnorm(convolved, is_training)
				layers.append(output)

	encoded = layers[-1]

	layer_specs = [
		(256, 0.0),    # decoder_6: [batch, 2, 2, 2, 256] => [batch, 4, 4, 4, 256]
		(256, 0.0),    # decoder_5: [batch, 4, 4, 4, 256] => [batch, 8, 8, 8, 256]
		(128, 0.0),    # decoder_4: [batch, 8, 8, 8, 256] => [batch, 16, 16, 16, 128]
		(64, 0.0),     # decoder_3: [batch, 16, 16, 16, 128] => [batch, 32, 32, 32, 64]
		(32, 0.0),     # decoder_2: [batch, 32, 32, 32, 64] => [batch, 64, 64, 64, 32]
	]    
	
	num_encoder_layers = len(layers)
	with tf.variable_scope('conv3d_autoencoder', reuse=reuse):
		for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
			skip_layer = num_encoder_layers - decoder_layer - 1
			with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
				if decoder_layer == 0:
					# first decoder layer doesn't have skip connections
					# since it is directly connected to the skip_layer
					input = layers[-1]
				else:
					# input = tf.concat([layers[-1], layers[skip_layer]], axis=4) # [batch, depth, height, width, in_channels] => [batch, depth, height, width, in_channels * 2]
					input = layers[-1]

				rectified = tf.nn.relu(input)
				output = gen_deconv3d(rectified, out_channels) # [batch, in_depth, in_height, in_width, in_channels] => [batch, in_depth*2, in_height*2, in_width*2, out_channels]
				output = batchnorm(output, is_training)
				if is_training and dropout > 0.0:
					output = tf.nn.dropout(output, keep_prob=1 - dropout)
				layers.append(output)

		# decoder_1: [batch, 64, 64, 64, 32] => [batch, 128, 128, 128, outchannel]
		with tf.variable_scope("decoder_1"):
			input = tf.concat([layers[-1], layers[0]], axis=4)
			rectified = tf.nn.relu(input)
			output = gen_deconv3d(rectified, outchannels)
			# decoded = tf.tanh(output)
			decoded = output
			layers.append(decoded)

	with tf.variable_scope('conv3d_autoencoder', reuse=reuse):
		# Loss function and optimizer
		with tf.variable_scope('loss_function'):
			loss = tf.reduce_mean(tf.square(data_y - decoded))
			data_y_occ 	= data_y[:,:,:,:,0]
			decoded_occ = decoded[:,:,:,:,0]
			loss_occ 	= tf.reduce_mean(tf.square(data_y_occ - decoded_occ))
			data_y_vel 	= data_y[:,:,:,:,1:4]
			decoded_vel = decoded[:,:,:,:,1:4]			
			loss_vel 	= tf.reduce_mean(tf.square(data_y_vel - decoded_vel))

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
				with tf.variable_scope('optimizer'):
					optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate,
														beta1=0.9,
														beta2=0.999,
														epsilon=1e-08).minimize(loss)

		return CONV3D_AE(
			decoded=decoded,
			loss=loss,
			loss_occ=loss_occ,
			loss_vel=loss_vel,
			optimizer=optimizer,
			encoded=encoded
		)

def main():
	#-- parse arguments
	model_restore = False  
	model_mode    ='train'  
	args = parse_args(model_restore, model_mode)
	# print msg to a file 
	printfile = open(os.path.join(args.log_dir, 'printlogfile'), 'w') 
	print(device_lib.list_local_devices(), flush=True, file=printfile)
	for key, value in args._get_kwargs():
		print(key, '=', value, flush=True, file=printfile)

	#-- load data addresss
	test_addrs_comp 		= load_data_addrs_comp()
	train_addrs, test_addrs = load_data_addrs(args, printfile)
	num_train_data  		= len(train_addrs)
	num_test_data   		= len(test_addrs)
	get_data_next_batch(train_addrs,0,20, False)
	# return

	#-- build convolutional autoencoder neural network
	data_x 		= tf.placeholder(tf.float32, shape=[args.batch_size, 128, 128, 128, 4], name='data_x')
	data_y 		= tf.placeholder(tf.float32, shape=[args.batch_size, 128, 128, 128, 4], name='data_y')
	AE_train	= conv3d_enc_dec(data_x=data_x, data_y=data_y, args=args, is_training=True, reuse=False)
	AE_Test     = conv3d_enc_dec(data_x=data_x, data_y=data_y, args=args, is_training=False, reuse=True)
	
	# summary placeholder
	with tf.name_scope('summary'):
		train_loss_protobuf 	= tf.summary.scalar('train_loss', AE_train.loss)
		train_loss_occ_protobuf = tf.summary.scalar('train_loss_occ', AE_train.loss_occ)
		train_loss_vel_protobuf = tf.summary.scalar('train_loss_vel', AE_train.loss_vel)
		test_loss_protobuf 		= tf.summary.scalar('test_loss', AE_Test.loss)
		test_loss_occ_protobuf 	= tf.summary.scalar('test_loss_occ', AE_Test.loss_occ)
		test_loss_vel_protobuf 	= tf.summary.scalar('test_loss_vel', AE_Test.loss_vel)

	# Initialize the variables (i.e. assign their default value)
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

	# 'Saver' op to save and restore all the variables
	checkpoints_saver = tf.train.Saver(max_to_keep=1)
	with tf.Session() as sess:
		summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
		# Run the initializer
		sess.run(init_op)

		# run training cycle
		for epoch in range(args.num_epoch):

			# run traing batches
			average_train_loss    	= 0
			average_train_loss_occ  = 0
			average_train_loss_vel  = 0
			num_train_batch     	= int(num_train_data/args.batch_size)			
			t_gen_data 				= 0
			t_train 				= 0
			for batch in range(num_train_batch):
				if batch % 200 == 0:
					print('(epoch, batch/num_train_batch) = ({}, {}/{})'.format(epoch, batch, num_train_batch),  flush=True, file=printfile)
				t_start 			= time.time()
				x_train, y_train 	= get_data_next_batch(train_addrs, batch, args.batch_size)
				t_end 	 			= time.time()
				t_gen_data 			+= t_end - t_start
				feed_dict   		= {data_x:x_train, data_y:y_train}
				fetch       		= { 'optimizer': AE_train.optimizer,
										'loss': AE_train.loss,
										'loss_occ': AE_train.loss_occ,
										'loss_vel': AE_train.loss_vel,
										'train_loss_summ': train_loss_protobuf,
										'train_loss_occ_summ': train_loss_occ_protobuf,
										'train_loss_vel_summ': train_loss_vel_protobuf }
				t_start 			= time.time()
				results        		= sess.run(fetches=fetch, feed_dict=feed_dict)
				t_end 				= time.time()
				t_train				+= t_end - t_start
				average_train_loss  += results['loss']
				average_train_loss_occ += results['loss_occ']
				average_train_loss_vel += results['loss_vel']
			print('train loss(epoch) = {:0.6f} * 4 = {:0.6f} + {:0.6f} * 3 ({})'.format(average_train_loss/num_train_batch, average_train_loss_occ/num_train_batch, average_train_loss_vel/num_train_batch, epoch),  flush=True, file=printfile)
			print('loading cost = {:0.6f}, training cost = {:0.6f}'.format(t_gen_data, t_train),  flush=True, file=printfile)
			summary_writer.add_summary(summary=results['train_loss_summ'], global_step=epoch)
			summary_writer.add_summary(summary=results['train_loss_occ_summ'], global_step=epoch)
			summary_writer.add_summary(summary=results['train_loss_vel_summ'], global_step=epoch)
			checkpoints_saver.save(sess, os.path.join(args.checkpoint_dir, 'model_{0}'.format(epoch)))         

			# run validation batches
			average_test_loss    			= 0
			average_test_loss_occ    		= 0
			average_test_loss_vel    		= 0
			num_test_batch     				= int(num_test_data/args.batch_size)				
			for batch in range(num_test_batch):
				x_test, y_test 				= get_data_next_batch(test_addrs, batch, args.batch_size)
				fetch   					= { 'loss': AE_Test.loss,
												'loss_occ': AE_Test.loss_occ,
												'loss_vel': AE_Test.loss_vel,
												'test_loss_summ': test_loss_protobuf,
												'test_loss_occ_summ': test_loss_occ_protobuf,
												'test_loss_vel_summ': test_loss_vel_protobuf }
				results						= sess.run(fetches=fetch, feed_dict={data_x:x_test, data_y:y_test})
				average_test_loss   		+= results['loss']
				average_test_loss_occ       += results['loss_occ']
				average_test_loss_vel       += results['loss_vel']
			print('test loss(epoch) = {:0.6f} * 4 = {:0.6f} + {:0.6f} * 3  ({})'.format(average_test_loss/num_test_batch, average_test_loss_occ/num_test_batch, average_test_loss_vel/num_test_batch, epoch),  flush=True, file=printfile)
			summary_writer.add_summary(summary=results['test_loss_summ'], global_step=epoch)
			summary_writer.add_summary(summary=results['test_loss_occ_summ'], global_step=epoch)
			summary_writer.add_summary(summary=results['test_loss_vel_summ'], global_step=epoch)

			# Test model: generate predict results
			if(epoch % 2 == 0 and epoch != 0):
				# for batch in range(50):
				# 	x_train, y_train= get_data_next_batch(train_addrs, batch, args.batch_size)
				# 	feed_dict   	= {data_x: x_train, data_y: y_train}
				# 	train_predict   = AE_Test.decoded.eval(session=sess, feed_dict=feed_dict)
				# 	np.save(os.path.join(args.result_dir,'train_input_{}.npy'.format(batch)), x_train)
				# 	np.save(os.path.join(args.result_dir,'train_output_{}.npy'.format(batch)), y_train)
				# 	np.save(os.path.join(args.result_dir,'train_predict_{}_{}.npy'.format(batch, epoch)), train_predict)

				for batch in range(50):			
					# compare unextend network predictions : use the unextend addresses
					x_test, y_test  = get_data_next_batch(test_addrs_comp, batch, args.batch_size)
					# x_test, y_test  = get_data_next_batch(test_addrs, batch, args.batch_size)
					feed_dict   	= {data_x: x_test, data_y: y_test}
					test_predict   	= AE_Test.decoded.eval(session=sess, feed_dict=feed_dict)
					np.save(os.path.join(args.result_dir,'test_input_{}.npy'.format(batch)), x_test)
					np.save(os.path.join(args.result_dir,'test_output_{}.npy'.format(batch)), y_test)
					np.save(os.path.join(args.result_dir,'test_predict_{}_{}.npy'.format(batch, epoch)), test_predict)
	printfile.close()

if __name__ == '__main__':
	main()