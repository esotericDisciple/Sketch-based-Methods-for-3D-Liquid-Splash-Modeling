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
header_example, content_example = uniio.readUni(os.path.join('../tensorflow/data', 'flipLevelSet_example.uni'))


def load_data_addrs(filepath, filetype):
	addrs 				= []
	addrs_sketch 		= []
	addrs_streamline 	= []
	addrs_lvst 			= []
	filename 			= []
	if filetype == 'train':
		filename = 'train_addrs.txt'
	else:
		filename = 'test_addrs.txt'
	with open(os.path.join(filepath, filename), 'r') as f:
		for line in f:
			(addr_x, addr_y) = line.split()
			addrs_sketch.append(addr_x)
			addrs_lvst.append(addr_y)
			addrs.append((addr_x, addr_y))
	
	print('num data = {:04d}'.format(len(addrs)))
	# print(test_addrs_sketch[0], test_addrs_lvst[0])

	return addrs_sketch, addrs_lvst

def get_data_addrs_next_batch(data_x_addrs, data_y_addrs, ibatch, batch_size):
	data_x = []
	data_y = []
	for i in range(ibatch*batch_size, ibatch*batch_size+batch_size):
		if os.path.isfile(data_x_addrs[i]) and os.path.isfile(data_y_addrs[i]):
			data_x.append(data_x_addrs[i])
			data_y.append(data_y_addrs[i])
		else:
			print('{:} or {:} does not exists!'.format(data_x_addrs[i], data_y_addrs[i]))	
	return data_x, data_y	

# visulize all files in result diretory
def surface_reconstruction_from_levelset(threshold_value, result_dir, mat_dir, manta_dir, ibatch, batch_size, iepoch, filetype, sketch_addrs, lvst_addrs):
	sketch_files 	= os.path.join(result_dir, '{}_input_{}.npy'.format(filetype, ibatch))
	gt_files 		= os.path.join(result_dir, '{}_output_{}.npy'.format(filetype, ibatch))
	predict_files 	= os.path.join(result_dir, '{}_predict_{}_{}.npy'.format(filetype, ibatch, iepoch))

	sketch_grid_data 		= np.load(sketch_files)
	gt_lvst_grid_data 		= np.load(gt_files)
	predict_lvst_grid_data 	= np.load(predict_files)
	# each .npy contains batch_size frames
	for i in range(batch_size):
		# the levelset grid is binarified during training
		gt_lvst 		= gt_lvst_grid_data[i]
		predict_lvst 	= predict_lvst_grid_data[i]

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
		# gt_lvst_copy 	= gt_lvst: Does not make a copy of gt_lvst, it merely creates a new reference to gt_lvst name!!!!
		gt_lvst_copy = np.copy(gt_lvst)
		gt_lvst_copy[gt_lvst<=threshold_value] = 0.5 # outside
		gt_lvst_copy[gt_lvst>threshold_value] = -0.5 # inside
		gt_lvst_copy = np.ascontiguousarray(gt_lvst_copy, dtype=np.float32)

		outpath_gt_lvst = os.path.join(result_dir,  seed_num+'#'+simname+'#'+filename_lvst+'_recons')
		uniio.writeUni(outpath_gt_lvst +'.uni', header_example, gt_lvst_copy)
		phi.load(outpath_gt_lvst +'.uni')		
		phi.createMesh(mesh)
		# for iters in range(3):
		# 	smoothMesh(mesh=mesh, strength=1e-3, steps=10) 
		# 	subdivideMesh(mesh=mesh, minAngle=0.01, minLength=0.5, maxLength=3*0.5, cutTubes=True)
		mesh.save(outpath_gt_lvst +'.obj')		
		os.remove(outpath_gt_lvst +'.uni')

		# prediction
		predict_lvst_copy = np.copy(predict_lvst)
		predict_lvst_copy[predict_lvst<=threshold_value] = 0.5 # outside
		predict_lvst_copy[predict_lvst>threshold_value] = -0.5 # inside
		predict_lvst_copy = np.ascontiguousarray(predict_lvst_copy, dtype=np.float32)

		outpath_pred_lvst = os.path.join(result_dir, seed_num+'#'+simname+'#'+filename_lvst+'_pred')
		# save current frame for update later
		np.save(outpath_pred_lvst+ '.npy', predict_lvst) 
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

	print('job: ibatch {:d} done'.format(ibatch))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Sketch-based Methods for 3D Liquid Splash Modeling')
	parser.add_argument('--batch_size', default=4, type=int)
	parser.add_argument('--num_batch', default=10, type=int)
	parser.add_argument('--iepoch', default=20, type=int)
	parser.add_argument('--threshold_value', default=0.6, type=float)
	parser.add_argument('--result_dir', default='../tensorflow/data/result', type=str)
	parser.add_argument('--mat_dir', default='../VisMat/data', type=str)
	parser.add_argument('--manta_dir', default='../manta/sbfs_scenes/data', type=str)
	parser.add_argument('--filetype', default='test', type=str)
	args = parser.parse_args()

	test_addrs_sketch, test_addrs_lvst = load_data_addrs(args.result_dir, args.filetype)
	for ibatch in range(args.num_batch):
		sketch_addrs, lvst_addrs  = get_data_addrs_next_batch(test_addrs_sketch, test_addrs_lvst, ibatch, args.batch_size)
		surface_reconstruction_from_levelset(args.threshold_value, args.result_dir, args.mat_dir, args.manta_dir, ibatch, args.batch_size, args.iepoch, args.filetype, sketch_addrs, lvst_addrs)