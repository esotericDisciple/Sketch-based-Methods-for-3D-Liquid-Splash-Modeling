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

# if GUI:
# 	gui = Gui()
# 	gui.show(dim==3)
# 	gui.setCamPos(0., 0., -2)
# 	gui.setCamRot(20,-60,0)

# uni head definition
header_example, content_example = uniio.readUni(os.path.join('../tensorflow/data_extend', 'flipLevelSet_example.uni'))

# load unextended version of addresses
def load_data_addrs_comp(filetype):
	addrs 				= []
	addrs_sketch 		= []
	addrs_lvst 			= []
	addrs_vel 			= []
	filename 			= []
	if filetype == 'train':
		filename = 'train_addrs.txt'
	else:
		filename = 'test_addrs.txt'
	with open(os.path.join('../tensorflow/data/result', filename), 'r') as f:
		for line in f:
			(addr_x, addr_y) = line.split()
			addrs_sketch.append(addr_x)
			addrs_lvst.append(addr_y)
			addr_lvst			= addr_y
			filename_lvst 		= addr_lvst[int(addr_lvst.rfind(os.sep)+1):int(addr_lvst.rfind('.'))]
			frame_num 			= filename_lvst[int(filename_lvst.rfind('_'))+1:]
			addr_lvst_pre 		= addr_lvst[0:int(addr_lvst.rfind(os.sep))]
			addr_lvst_pre_pre 	= addr_lvst_pre[0:int(addr_lvst_pre.rfind(os.sep))]
			simname	 			= addr_lvst_pre[int(addr_lvst_pre.rfind(os.sep)+1):]
			addr_z 				= addr_lvst_pre + os.sep + 'flipVel_' + frame_num + '.uni'
			addrs_vel.append(addr_z)
			addrs.append((addr_x, addr_y, addr_z))
	
	print('num data = {:04d}'.format(len(addrs)))
	# print(test_addrs_sketch[0], test_addrs_lvst[0])

	return addrs_sketch, addrs_lvst, addrs_vel	

# ../VisMat/data\4096\sbfs_flip_water_pipes_10113\flipSketchGrid_0030.bin 
# ../manta/sbfs_scenes/data\sbfs_flip_water_pipes_10113\flipLevelSet_0030.uni 
# ../manta/sbfs_scenes/data\sbfs_flip_water_pipes_10113\flipVel_0030.uni
def load_data_addrs(filetype):
	addrs 				= []
	addrs_sketch 		= []
	addrs_lvst 			= []
	addrs_vel 			= []
	filename 			= []
	if filetype == 'train':
		filename = 'train_addrs.txt'
	else:
		filename = 'test_addrs.txt'
	with open(os.path.join('./data_extend/result', filename), 'r') as f:
		for line in f:
			(addr_x, addr_y, addr_z) = line.split()
			addrs_sketch.append(addr_x)
			addrs_lvst.append(addr_y)
			addrs_vel.append(addr_z)
			addrs.append((addr_x, addr_y, addr_z))
	
	print('num data = {:04d}'.format(len(addrs)))
	# print(test_addrs_sketch[0], test_addrs_lvst[0])

	return addrs_sketch, addrs_lvst, addrs_vel

def get_data_addrs_next_batch(data_x_addrs, data_y_addrs, data_z_addrs, ibatch, batch_size):
	data_x = []
	data_y = []
	data_z = []
	for i in range(ibatch*batch_size, ibatch*batch_size+batch_size):
		if os.path.isfile(data_x_addrs[i]) and os.path.isfile(data_y_addrs[i]) and os.path.isfile(data_z_addrs[i]):
			data_x.append(data_x_addrs[i])
			data_y.append(data_y_addrs[i])
			data_z.append(data_z_addrs[i])
		else:
			print('{:}, {:} or {:} does not exists!'.format(data_x_addrs[i], data_y_addrs[i], data_z_addrs[i]))	
	return data_x, data_y, data_z	

# visulize all files in result diretory
def surface_reconstruction_from_levelset(threshold_value, result_dir, mat_dir, manta_dir, batch, batch_size, iepoch, filetype, sketch_addrs, lvst_addrs, vel_addrs):
	sketch_file 		= os.path.join(result_dir, '{}_input_{}.npy'.format(filetype, batch))
	gt_file  			= os.path.join(result_dir, '{}_output_{}.npy'.format(filetype, batch))
	predict_file 		= os.path.join(result_dir, '{}_predict_{}_{}.npy'.format(filetype, batch, iepoch))

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
		# save current frame for update later
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

		src_mesh = os.path.join(manta_dir, simname, 'flip_'+frame_num+'.gz')
		dst_mesh = os.path.join(result_dir, simname+'#'+'flip_'+frame_num+'.gz')
		copyfile(src_mesh, dst_mesh)

	print('job: batch {:d} done'.format(batch))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Sketch-based Methods for 3D Liquid Splash Modeling')
	parser.add_argument('--batch_size', default=4, type=int)
	parser.add_argument('--num_batch', default=10, type=int)
	parser.add_argument('--iepoch', default=20, type=int)
	parser.add_argument('--threshold_value', default=0.6, type=float)
	parser.add_argument('--result_dir', default='../tensorflow/data_extend/result', type=str)
	parser.add_argument('--mat_dir', default='../VisMat/data', type=str)
	parser.add_argument('--manta_dir', default='../manta/sbfs_scenes/data', type=str)
	parser.add_argument('--filetype', default='test', type=str)
	args = parser.parse_args()

	# compare unextend network predictions : use the unextend addresses
	test_addrs_sketch, test_addrs_lvst, test_addrs_vel = load_data_addrs_comp(args.filetype)
	# test_addrs_sketch, test_addrs_lvst, test_addrs_vel = load_data_addrs(args.filetype)
	for ibatch in range(args.num_batch):
		sketch_addrs, lvst_addrs, vel_addrs  = get_data_addrs_next_batch(test_addrs_sketch, test_addrs_lvst, test_addrs_vel, ibatch, args.batch_size)
		surface_reconstruction_from_levelset(args.threshold_value, args.result_dir, args.mat_dir, args.manta_dir, ibatch, args.batch_size, args.iepoch, args.filetype, sketch_addrs, lvst_addrs, vel_addrs)