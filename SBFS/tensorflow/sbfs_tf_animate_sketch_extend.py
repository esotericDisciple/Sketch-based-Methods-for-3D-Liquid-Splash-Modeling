from manta import *
import uniio

import argparse
import numpy as np 
from shutil import copyfile
import os
import collections
import glob

#-- parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dim', default=3, type=int, help='domain dimension')
parser.add_argument('--res', default=128, type=int, help='domain resolution')
parser.add_argument('--gravity', default=-0.0005, type=float, help='gravity')
parser.add_argument('--visc', default=0.000, type=float, help='viscosity')
parser.add_argument('--surfaceTension', default=0.0, type=float, help='surfaceTension')
parser.add_argument('--flipRatio', default=0.97, type=float, help='flipRatio')
parser.add_argument('--accuracy', default=1e-4, type=float, help='cgAccuracy')

args = parser.parse_args()

#-- save simulation data
saveData  	= True
baseDir  	= 'demo/'
sceneName	= 'sbfs_flip_water_pipes_'
simNo     	= 10000 # simulation scene id
simDir 		= ''    # path to folder 
steps   	= 100   # number of simulation steps

#-- mantaflow fluid solver
res = 128
dim = 3
gs 	= vec3(res,res,res)
s 	= Solver(name='main', gridSize = gs, dim=dim)
# Adaptive time stepping
s.frameLength = 0.8				 # length of one frame (in "world time")
s.cfl         = 3.0				 # maximal velocity per cell and timestep, 3 is fairly strict
s.timestep    = s.frameLength
s.timestepMin = s.frameLength / 4.  # time step range
s.timestepMax = s.frameLength * 4.

particleNumber = 2
randomness     = 0.1
minParticles   = pow(2,3)

#-- Prepare grids and particles
flags    		= s.create(FlagGrid)

phi      		= s.create(LevelsetGrid) 
phiFluidIn   	= s.create(LevelsetGrid)  
phiObs    		= s.create(LevelsetGrid)
phiObsIn    	= s.create(LevelsetGrid)

phiMesh_obs   = s.create(LevelsetGrid)
phiMesh_fluid = s.create(LevelsetGrid)

pressure 	= s.create(RealGrid)
fractions 	= s.create(MACGrid)
curv     	= s.create(RealGrid)

vel      	= s.create(MACGrid)
vel_fluid   = s.create(MACGrid)
velOld   	= s.create(MACGrid)
velParts 	= s.create(MACGrid)
mapWeights  = s.create(MACGrid)

pp        		= s.create(BasicParticleSystem) 
pVel      		= pp.create(PdataVec3)
mesh  			= s.create(Mesh)
mesh_fluid  	= s.create(Mesh)
mesh_fluid_vel  = s.create(Mesh)
mesh_obs    	= s.create(Mesh)

pindex 			= s.create(ParticleIndexSystem)
gpi    			= s.create(IntGrid)

# uni head definition
header_example, content_example = uniio.readUni(os.path.join('../tensorflow/data_extend', 'flipLevelSet_example.uni'))
header_vel_example, content_vel_example = uniio.readUni(os.path.join('../tensorflow/data_extend', 'flipVel_example.uni'))

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

def get_data_addrs_cur_frame(data_x_addrs, data_y_addrs, data_z_addrs, cur_frame):
	if os.path.isfile(data_x_addrs[cur_frame]) and os.path.isfile(data_y_addrs[cur_frame]) and os.path.isfile(data_z_addrs[cur_frame]):
		return data_x_addrs[cur_frame], data_y_addrs[cur_frame], data_z_addrs[cur_frame]
	else:
		print('{:}, {} or {:} does not exists!'.format(data_x_addrs[cur_frame], data_y_addrs[cur_frame], data_z_addrs[cur_frame]))

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

	# load the original predict levelset and velocity
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

def flip(t):
	#** FLIP ALGORITHM

	#**1: Advect particles
	pp.advectInGrid(flags=flags, vel=vel, integrationMode=IntRK4, deleteInObstacle=False )

	#**2.1: Map particles to grid(velocity)
	mapPartsToMAC(vel=vel, flags=flags, velOld=velOld, parts=pp, partVel=pVel, weight=mapWeights) # velOld is stored
	extrapolateMACFromWeight( vel=vel , distance=2, weight=mapWeights ) # make sure we have velocities throught liquid region
	markFluidCells( parts=pp, flags=flags )								# mark all particles in flaggrid as fluid

	#**2.2: Map particles to grid(levelset)
	gridParticleIndex( parts=pp , flags=flags, indexSys=pindex, index=gpi )
	unionParticleLevelset( pp, pindex, flags, gpi, phi ) 
	resetOutflow(flags=flags,parts=pp,index=gpi,indexSys=pindex)        # delete fluid and ensure empty flag in outflow cells, delete particles and density and set phi to 0.5
	extrapolateLsSimple(phi=phi, distance=4, inside=True);              # simple extrapolation functions for levelsets

	#**3: Forces & pressure solve
	addGravity(flags=flags, vel=vel, gravity=(0,args.gravity,0))

	# vel diffusion / viscosity!
	# diffusion param for solve = const * dt / dx^2
	setWallBcs(flags=flags, vel=vel)
	cgSolveDiffusion( flags, vel, args.visc * s.timestep * float(args.res*args.res) )

	setWallBcs(flags=flags, vel=vel)	
	getLaplacian(laplacian=curv, grid=phi)  # surface tension helper
	solvePressure(flags=flags, vel=vel, pressure=pressure, phi=phi, curv=curv, surfTens=args.surfaceTension, cgAccuracy=args.accuracy)
	setWallBcs(flags=flags, vel=vel)

	extrapolateMACSimple( flags=flags, vel=vel)
	
	# optional: set source grids for resampling, used in adjustNumber!
	pVel.setSource( vel, isMAC=True )
	adjustNumber( parts=pp, vel=vel, flags=flags, minParticles=1*minParticles, maxParticles=2*minParticles, phi=phi) 

	#**4: Update particle velocities
	flipVelocityUpdate(vel=vel, velOld=velOld, flags=flags, parts=pp, partVel=pVel, flipRatio=args.flipRatio )

	# post-processing surfaceTurbulence.py
	# particleSurfaceTurbulence( flags=flags, coarseParts=pp, coarsePartsPrevPos=pPrevPos, surfPoints=surfacePoints, surfaceNormals=surfaceNormal, surfaceWaveH=surfaceWaveH, surfaceWaveDtH=surfaceWaveDtH,surfacePointsDisplaced=surfacePointsDisplaced, surfaceWaveSource=surfaceWaveSource, surfaceWaveSeed=surfaceWaveSeed, surfaceWaveSeedAmplitude=surfaceWaveSeedAmplitude, res=args.res,
	# 	nbSurfaceMaintenanceIterations = 6,
	# 	surfaceDensity = 12,
	# 	outerRadius = 1.0*radiusFactor,
	# 	dt = 0.005,
	# 	waveSpeed = 32, # res,
	# 	waveDamping = 0.05,
	# 	waveSeedFrequency = 4.0,# res/8.0,
	# 	waveMaxAmplitude = 0.5, # res/64.0
	# 	waveMaxSeedingAmplitude = 0.5, # as a multiple of max amplitude
	# 	waveMaxFrequency = 128.0,
	# 	waveSeedingCurvatureThresholdRegionCenter = 0.025, # any curvature higher than this value will seed waves
	# 	waveSeedingCurvatureThresholdRegionRadius = 0.01,
	# 	waveSeedStepSizeRatioOfMax = 0.05 # higher values will result in faster and more violent wave seeding
	# 	)

def manta_animate_from_reconstruction(result_dir, mat_dir, manta_dir, sketch_addr, lvst_addr, vel_addr, cur_frame, doOpen):
	outdir = './demo/' + '%04d/' % cur_frame
	if not os.path.exists(outdir):
		os.makedirs(outdir)

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

	# copy the streamline and original mesh
	src_line = os.path.join(mat_dir, seed_num, simname, 'flipStreamline_'+frame_num+'_resampled.txt')
	dst_line = os.path.join(outdir, seed_num+'#'+simname+'#'+'flipStreamline_'+frame_num+'_resampled.txt')
	copyfile(src_line, dst_line)

	src_mesh = os.path.join(manta_dir, simname, 'flip_'+frame_num+'.gz')
	dst_mesh = os.path.join(outdir, simname+'#'+'flip_'+frame_num+'.gz')
	copyfile(src_mesh, dst_mesh)

	# load the predict levelset and velocity
	outpath_pred_lvst = os.path.join(result_dir, seed_num+'#'+simname+'#'+filename_lvst+'_pred')
	predict_vec = np.load(outpath_pred_lvst+ '.npy')
	predict_occ = predict_vec[:,:,:,0]
	predict_vel = predict_vec[:,:,:,1:4]


	# mantaflow fluid simulation here
	global flags, phi, mesh_fluid, phiFluidIn, vel

	bWidth=1
	flags.initDomain(boundaryWidth=bWidth, phiWalls=phiObs)
	if doOpen:
		setOpenBound(flags, bWidth,'xXyYzZ',FlagOutflow|FlagEmpty) 
	phi.initFromFlags(flags)

    # load fluid volume (obj file from reconstruction)
	mesh_fluid.load(outpath_pred_lvst +'.obj')
	print('mesh {} loaded successfully'.format(outpath_pred_lvst +'.obj'))
	mesh_fluid.scale(vec3(args.res, args.res, args.res))
	mesh_fluid.offset(vec3(args.res, args.res, args.res)*0.5)
	mesh_fluid.computeLevelset(phiFluidIn, 2.)
	
	phi.join(phiFluidIn)

 	# load the fluid velocity
	predict_vel_copy = np.copy(predict_vel)
	predict_vel_copy = np.ascontiguousarray(predict_vel_copy, dtype=np.float32)
	uniio.writeUni(outdir + 'vel.uni', header_vel_example, predict_vel_copy)
	vel_fluid.load(outdir + 'vel.uni')

	vel.copyFrom(vel_fluid)

	global pp, pvel, s, steps
	# update flag and sample particles
	flags.updateFromLevelset(phi)
	sampleLevelsetWithParticles( phi=phi, flags=flags, parts=pp, discretization=2, randomness=0.05 )
	mapGridToPartsVec3(source=vel, parts=pp, target=pVel )

	if GUI:
		gui = Gui()
		gui.show()
		gui.setCamPos(0., 0., -2)
		gui.setCamRot(0,0,0)
		# gui.pause()
		
	for t in range(steps):
		maxVel = vel.getMax()
		s.adaptTimestep( maxVel )
		mantaMsg('\nFrame %i, timestep %f, timeTotal %f' % (s.frame, s.timestep, s.timeTotal))
		
		flip(t)
		s.step()

		phi.createMesh(mesh)
		if saveData:
			# vel.save(outdir + 'flipVel_%04d.uni' % t)
			# pp.save( outdir + 'flipParts_%04d.uni' % t )
			# phi.save(outdir + 'flipLevelSet_%04d.uni' % t)
			mesh.save(outdir + 'flip_%04d.gz' % t )
			gui.screenshot( outdir + 'flip_%04d.png' % t )

if __name__ == '__main__':
	cur_frame      	= 15
	doOpen 			= False
	batch_size 		= 4
	num_batch   	= 10
	iepoch  		= 20
	threshold_value = 0.6
	result_dir 		= './data_extend/result'
	mat_dir 		= '../VisMat/data'
	manta_dir 		= '../manta/sbfs_scenes/data' 
	filetype    	= 'test'

	# compare unextend network predictions : use the unextend addresses
	test_addrs_sketch, test_addrs_lvst, test_addrs_vel = load_data_addrs_comp(filetype)
	# test_addrs_sketch, test_addrs_lvst, test_addrs_vel = load_data_addrs(filetype)

	sketch_addr, lvst_addr, vel_addr = get_data_addrs_cur_frame(test_addrs_sketch, test_addrs_lvst, test_addrs_vel, cur_frame)
	manta_animate_from_reconstruction(result_dir, mat_dir, manta_dir, sketch_addr, lvst_addr, vel_addr, cur_frame, doOpen)

