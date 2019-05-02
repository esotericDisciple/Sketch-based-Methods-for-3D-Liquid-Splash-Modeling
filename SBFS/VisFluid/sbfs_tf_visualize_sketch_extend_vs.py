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

	print('num traing data = {:04d}'.format(len(train_addrs)))
	print('num testing data = {:04d}'.format(len(test_addrs)))	
	
	return train_addrs, test_addrs

# visulize all files in result diretory
def surface_reconstruction_from_levelset(iepoch, threshold_value, result_dir, mat_dir, manta_dir, batch, batch_size, sketch_addrs, lvst_addrs, vel_addrs):

	sketch_file 		= os.path.join(result_dir, 'test_input_{}_{}.npy'.format(batch, iepoch))
	gt_file  			= os.path.join(result_dir, 'test_output_{}_{}.npy'.format(batch, iepoch))
	predict_file 		= os.path.join(result_dir, 'test_predict_{}_{}.npy'.format(batch, iepoch))

	input_data 		= np.load(sketch_file)
	gt_data 		= np.load(gt_file)
	predict_data 	= np.load(predict_file)
	# each .npy contains batch_size frames
	for i in range(batch_size):	
		gt_vec	 	= gt_data[i]
		predict_vec = predict_data[i]

		gt_occ 		= gt_vec[:,:,:,0]
		gtt_vel 	= gt_vec[:,:,:,1:4]

		predict_occ = predict_vec[:,:,:,0]
		predict_vel = predict_vec[:,:,:,1:4]		

		# the corresponding filename and sim folder
		addr_lvst 			= lvst_addrs[i]
		filename_lvst 		= addr_lvst[int(addr_lvst.rfind(os.sep)+1):int(addr_lvst.rfind('.'))]
		frame_num 			= filename_lvst[int(filename_lvst.rfind('_'))+1:]
		addr_lvst_pre 		= addr_lvst[0:int(addr_lvst.rfind(os.sep))]
		addr_lvst_pre_pre 	= addr_lvst_pre[0:int(addr_lvst_pre.rfind(os.sep))]
		simname	 			= addr_lvst_pre[int(addr_lvst_pre.rfind(os.sep)+1):]

		addr_sketch 		= sketch_addrs[i]
		addr_sketch_pre     = addr_sketch[0:int(addr_sketch.rfind(os.sep))]
		addr_sketch_pre_pre = addr_sketch_pre[0:int(addr_sketch_pre.rfind(os.sep))]
		seed_num 			= addr_sketch_pre_pre[int(addr_sketch_pre_pre.rfind(os.sep)+1):]
		# print(seed_num, simname, filename_lvst, frame_num)

		# groundtruth
		gt_occ_copy 	= np.copy(gt_occ)
		gt_occ_copy[gt_occ<=threshold_value] = 0.5 # outside
		gt_occ_copy[gt_occ>threshold_value] = -0.5 # inside
		gt_occ_copy = np.ascontiguousarray(gt_occ_copy, dtype=np.float32)
		
		outpath_gt_lvst = os.path.join(result_dir,  seed_num+'#'+simname+'#'+filename_lvst+'_recons')
		uniio.writeUni(outpath_gt_lvst +'.uni', header_example, gt_occ_copy)
		phi.load(outpath_gt_lvst +'.uni')		
		phi.createMesh(mesh)
		# for iters in range(3):
		# 	smoothMesh(mesh=mesh, strength=1e-3, steps=10) 
		# 	subdivideMesh(mesh=mesh, minAngle=0.01, minLength=0.5, maxLength=3*0.5, cutTubes=True)
		mesh.save(outpath_gt_lvst +'.obj')		
		os.remove(outpath_gt_lvst +'.uni')

		# prediction
		predict_lvst_copy = np.copy(predict_occ)
		predict_lvst_copy[predict_occ<=threshold_value] = 0.5 # outside
		predict_lvst_copy[predict_occ>threshold_value] = -0.5 # inside
		predict_lvst_copy = np.ascontiguousarray(predict_lvst_copy, dtype=np.float32)

		outpath_pred_lvst = os.path.join(result_dir, seed_num+'#'+simname+'#'+filename_lvst+'_pred')
		# save the npy file(data) as the update reference
		np.save(outpath_pred_lvst+ '.npy', predict_vec) 
		uniio.writeUni(outpath_pred_lvst +'.uni', header_example, predict_lvst_copy)
		phi.load(outpath_pred_lvst +'.uni')		
		phi.createMesh(mesh)
		# for iters in range(3):
		# 	smoothMesh(mesh=mesh, strength=1e-3, steps=10) 
		# 	subdivideMesh(mesh=mesh, minAngle=0.01, minLength=0.5, maxLength=3*0.5, cutTubes=True)
		mesh.save(outpath_pred_lvst +'.obj')		
		os.remove(outpath_pred_lvst +'.uni')

		# the streamline and original mesh
		src_line = os.path.join(mat_dir, seed_num, simname, 'flipStreamline_'+frame_num+'_resampled.txt')
		dst_line = os.path.join(result_dir, seed_num+'#'+simname+'#'+'flipStreamline_'+frame_num+'_resampled.txt')
		copyfile(src_line, dst_line)

		src_line = os.path.join(mat_dir, seed_num, simname, 'flipStreamline_'+frame_num+'_resampled_agglomerativeclustering_cluster_centers.txt')
		dst_line = os.path.join(result_dir, seed_num+'#'+simname+'#'+'flipStreamline_'+frame_num+'_resampled_agglomerativeclustering_cluster_centers.txt')
		copyfile(src_line, dst_line)

		src_mesh = os.path.join(manta_dir, simname, 'flip_'+frame_num+'.gz')
		dst_mesh = os.path.join(result_dir, simname+'#'+'flip_'+frame_num+'.gz')
		copyfile(src_mesh, dst_mesh)

	print('job: batch {:d} done'.format(batch))

def visualize_evaluation(args):

	data_train_addrs, data_test_addrs = load_data_addrs(args)
	num_test_data 	= len(data_test_addrs)
	num_test_batch  = int(num_test_data/args.batch_size)
	for batch in range(args.num_batch):	
		sketch_addrs 	= []
		lvst_addrs 		= []
		vel_addrs 		= []
		for i in range(batch*args.batch_size, batch*args.batch_size+args.batch_size):
			(data_sketch_addr, data_lvst_addr, data_vel_addr) = data_test_addrs[i]
			if os.path.isfile(data_sketch_addr) and os.path.isfile(data_lvst_addr) and os.path.isfile(data_vel_addr):
				sketch_addrs.append(data_sketch_addr)
				lvst_addrs.append(data_lvst_addr)
				vel_addrs.append(data_vel_addr)
			else:
				print('{:}, {:} or {:} does not exists!'.format(data_sketch_addr, data_lvst_addr, data_vel_addr))	
		surface_reconstruction_from_levelset(args.iepoch, args.threshold_value, args.result_dir, args.mat_dir, args.manta_dir, batch, args.batch_size, sketch_addrs, lvst_addrs, vel_addrs)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Sketch-based Methods for 3D Liquid Splash Modeling')
	parser.add_argument('--batch_size', default=4, type=int)
	parser.add_argument('--num_batch', default=10, type=int)
	parser.add_argument('--iepoch', default=16, type=int)
	parser.add_argument('--threshold_value', default=0.6, type=float)
	parser.add_argument('--result_dir', default='../tensorflow/data_extend_REALGAN_cluster/result', type=str)
	parser.add_argument('--mat_dir', default='../VisMat/data', type=str)
	parser.add_argument('--manta_dir', default='../manta/sbfs_scenes/data', type=str)
	parser.add_argument('--shared_dir', default='../tensorflow/shared', type=str)
	args = parser.parse_args()
	
	visualize_evaluation(args)

