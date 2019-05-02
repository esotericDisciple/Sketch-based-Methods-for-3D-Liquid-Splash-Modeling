from manta import *
import uniio

import argparse
import numpy as np 
from shutil import copyfile
import os
import collections
import glob


#-- mantaflow fluid solver
res = 128
dim = 3
gs 	= vec3(res,res,res)
s 	= Solver(name='main', gridSize = gs, dim=dim)

phi      = s.create(LevelsetGrid)
mesh     = s.create(Mesh)
flags    = s.create(FlagGrid)
flags.initDomain(boundaryWidth=0)

# uni head definition
header_example, content_example = uniio.readUni(os.path.join('../tensorflow/shared', 'flipLevelSet_example.uni'))

# ../VisMat/data\4096\sbfs_flip_water_pipes_10113\flipSketchGrid_0030.bin 
# ../manta/sbfs_scenes/data\sbfs_flip_water_pipes_10113\flipLevelSet_0030.uni 
# ../manta/sbfs_scenes/data\sbfs_flip_water_pipes_10113\flipVel_0030.uni
def load_data_addrs(args):
	addrs = []
	train_addrs = []
	test_addrs = []

	with open(os.path.join(args.shared_dir, 'train_addrs.txt'), 'r') as f:
		for line in f:
			(addr_x, addr_y, addr_z)= line.split()
			train_addrs.append((addr_x, addr_y, addr_z))
	with open(os.path.join(args.shared_dir, 'test_addrs.txt'), 'r') as f:
		for line in f:
			(addr_x, addr_y, addr_z)= line.split()
			test_addrs.append((addr_x, addr_y, addr_z))

	# print('num traing data = {:04d}'.format(len(train_addrs)))
	# print('num testing data = {:04d}'.format(len(test_addrs)))	
	
	return train_addrs, test_addrs

def get_data_addrs_update_frame(data_addrs, update_frame):
	(data_sketch_addr, data_lvst_addr, data_vel_addr) = data_addrs[update_frame]
	if os.path.isfile(data_sketch_addr) and os.path.isfile(data_lvst_addr) and os.path.isfile(data_vel_addr):
		return data_sketch_addr, data_lvst_addr, data_vel_addr
	else:
		print('{:}, {} or {:} does not exists!'.format(data_sketch_addr, data_lvst_addr,data_vel_addr))

def surface_reconstruction_from_levelset_update(threshold_value, result_dir, mat_dir, manta_dir, sketch_addr, lvst_addr, vel_addr):
	addr_lvst 			= lvst_addr
	filename_lvst 		= addr_lvst[int(addr_lvst.rfind(os.sep)+1):int(addr_lvst.rfind('.'))]
	frame_num 			= filename_lvst[int(filename_lvst.rfind('_'))+1:]
	addr_lvst_pre 		= addr_lvst[0:int(addr_lvst.rfind(os.sep))]
	addr_lvst_pre_pre 	= addr_lvst_pre[0:int(addr_lvst_pre.rfind(os.sep))]
	simname	 			= addr_lvst_pre[int(addr_lvst_pre.rfind(os.sep)+1):]

	addr_sketch 		= sketch_addr
	addr_sketch_pre     = addr_sketch[0:int(addr_sketch.rfind(os.sep))]
	addr_sketch_pre_pre = addr_sketch_pre[0:int(addr_sketch_pre.rfind(os.sep))]
	seed_num 			= addr_sketch_pre_pre[int(addr_sketch_pre_pre.rfind(os.sep)+1):]	

	# load the original predict data: levelset and velocity
	outpath_pred_lvst = os.path.join(result_dir, seed_num+'#'+simname+'#'+filename_lvst+'_pred')
	predict_vec = np.load(outpath_pred_lvst+ '.npy')

	predict_occ = predict_vec[:,:,:,0]
	predict_vel = predict_vec[:,:,:,1:4]

	predict_lvst_copy = np.copy(predict_occ)
	predict_lvst_copy[predict_occ<=threshold_value] = 0.5 # outside
	predict_lvst_copy[predict_occ>threshold_value] = -0.5 # inside
	predict_lvst_copy = np.ascontiguousarray(predict_lvst_copy, dtype=np.float32)

	uniio.writeUni(outpath_pred_lvst +'.uni', header_example, predict_lvst_copy)
	phi.load(outpath_pred_lvst +'.uni')		
	phi.createMesh(mesh)
	# for iters in range(3):
	# 	smoothMesh(mesh=mesh, strength=1e-3, steps=10) 
	# 	subdivideMesh(mesh=mesh, minAngle=0.01, minLength=0.5, maxLength=3*0.5, cutTubes=True)
	mesh.save(outpath_pred_lvst +'.obj')		
	os.remove(outpath_pred_lvst +'.uni')

def visualize_evaluation_update(args):

	data_train_addrs, data_test_addrs = load_data_addrs(args)
	num_test_data 	= len(data_test_addrs)
	num_test_batch  = int(num_test_data/args.batch_size)

	sketch_addr, lvst_addr, vel_addr = get_data_addrs_update_frame(data_test_addrs, args.update_frame)
	surface_reconstruction_from_levelset_update(args.threshold_value, args.result_dir, args.mat_dir, args.manta_dir, sketch_addr, lvst_addr, vel_addr)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Sketch-based Methods for 3D Liquid Splash Modeling')
	parser.add_argument('--update_frame', required=True, type=int)
	parser.add_argument('--batch_size', default=4, type=int)
	parser.add_argument('--num_batch', default=10, type=int)
	parser.add_argument('--iepoch', default=16, type=int)
	parser.add_argument('--threshold_value', default=0.6, type=float)
	parser.add_argument('--result_dir', default='../tensorflow/data_extend_REALGAN_cluster/result', type=str)
	parser.add_argument('--mat_dir', default='../VisMat/data', type=str)
	parser.add_argument('--manta_dir', default='../manta/sbfs_scenes/data', type=str)
	parser.add_argument('--shared_dir', default='../tensorflow/shared', type=str)
	args = parser.parse_args()

	visualize_evaluation_update(args)

