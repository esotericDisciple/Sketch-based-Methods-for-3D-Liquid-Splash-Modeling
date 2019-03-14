#
# data generation for sketch based fluid simulation: pure flip, no resampling
# 
from manta import *

import numpy as np

import os
import json
import random
import math
import argparse
import sys

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
for k, v in args._get_kwargs():
	print(k, "=", v)

#-- save simulation data
saveData  	= True
baseDir  	= 'data/'
sceneName	= 'sbfs_flip_water_pipes_'
simNo     	= 10000 # simulation scene id
simDir 		= ''    # path to folder 
steps   	= 120   # number of simulation steps


#-- create the fluid solver
gs = vec3(args.res,args.res,args.res)
s = Solver(name='main', gridSize = gs, dim=args.dim)
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
velOld   	= s.create(MACGrid)
velParts 	= s.create(MACGrid)
mapWeights  = s.create(MACGrid)

pp        		= s.create(BasicParticleSystem) 
pVel      		= pp.create(PdataVec3)
mesh_fluid  	= s.create(Mesh)
mesh_fluid_vel  = s.create(Mesh)
mesh_obs    	= s.create(Mesh)

pindex 			= s.create(ParticleIndexSystem)
gpi    			= s.create(IntGrid)

surfacePointsDisplaced = s.create(BasicParticleSystem)
spdDummy = surfacePointsDisplaced.create(PdataVec3) # dummy for display
pPrevPos  = pp.create(PdataVec3)
surfacePoints = s.create(BasicParticleSystem)
surfaceNormal = surfacePoints.create(PdataVec3)
surfaceWaveH = surfacePoints.create(PdataReal)
surfaceWaveDtH = surfacePoints.create(PdataReal)
surfaceWaveSource = surfacePoints.create(PdataReal)
surfaceWaveSeedAmplitude = surfacePoints.create(PdataReal)
surfaceWaveSeed = surfacePoints.create(PdataReal)
radiusFactor = 1.0

##-- scene setup
seed = random.randint(0, 2**31 - 1)
np.random.seed(seed)

bWidth=1 
doOpen = True

altitude 	= 90 / 360 * 2 * 3.1415926  # flow angle
azimuth 	= 0 / 360 * 2 * 3.1415926   # flow angle
n_x 		= 2							# number of flow
n_z 		= 2							# number of flow
p_center 	= vec3(0.5,0.2,0.5)			# center posistion
d_x 		= 0.02						# interval of flows
d_z 		= 0.05						# interval of flows

nn_x 		= 1							# number of overlapped flows
nn_z 		= 5 						# number of overlapped flows
dd_x 		= 0.01						# overlapped flows
dd_z 		= 0.01						# overlapped flows
radius 		= 1  						# flow source size  

d_alt  = np.random.uniform(-45 / 360 * 2 * 3.1415926, 45 / 360 * 2 * 3.1415926, size=(2*n_x, 2*n_z))  # flow angle variance
d_azi  = np.random.uniform(-180 / 360 * 2 * 3.1415926, 180 / 360 * 2 * 3.1415926, size=(2*n_x, 2*n_z))  # flow angle variance

# inflow start and duration time
t_start    = np.random.randint(0, 21, size=(2*n_x, 2*n_z))
t_duration = np.random.randint(20, 21, size=(2*n_x, 2*n_z))

# fluid velocity and turn on
fluidVelMag  = np.random.uniform(3.0, 3.0, size=(2*n_x, 2*n_z))
fluidSwitch  = np.random.choice(a=[False, True], size=(2*n_x, 2*n_z), p=[0.5, 0.5])

print('inflow start time')
print(t_start)
print(fluidSwitch)

# save scene configuration parameters
def save_scene_config():
	global args, d_alt, d_azi, t_start, t_duration, fluidVelMag, fluidSwitch
	global simNo, simDir

	if  saveData:
		# create next unused folder
		folderNo = simNo
		pathaddition = sceneName + '%04d/' % folderNo
		while os.path.exists(baseDir + pathaddition):
			folderNo += 1
			pathaddition = sceneName + '%04d/' % folderNo
		simDir = baseDir + pathaddition	
		simNo = folderNo
		os.makedirs(simDir)

		# save args options
		with open(os.path.join(simDir, "options.json"), "w") as f:
			f.write(json.dumps(vars(args), sort_keys=True, indent=4))
		
		pressure.save( simDir + 'ref_flipParts_0000.uni' )

		np.save(simDir + 'd_alt.npy', d_alt)
		np.save(simDir + 'd_azi.npy', d_azi)
		np.save(simDir + 't_start.npy', t_start)
		np.save(simDir + 't_duration.npy', t_duration)
		np.save(simDir + 'fluidVelMag.npy', fluidVelMag)
		np.save(simDir + 'fluidSwitch.npy', fluidSwitch)

# load scene configuration from previous simulation
def load_scene_config():
	global args, d_alt, d_azi, t_start, t_duration, fluidVelMag, fluidSwitch
	global simNo, simDir

	# load from reference scene folder
	refSimNo  = 10100
	refSimDir = baseDir + sceneName + '%04d/' % refSimNo
	with open(os.path.join(refSimDir, "options.json")) as f:
		json_dict = json.load(f)
		argparse_dict = vars(args)
		argparse_dict.update(json_dict)
	d_alt 		= np.load(refSimDir + 'd_alt.npy')
	d_azi 		= np.load(refSimDir +'d_azi.npy')
	t_start 	= np.load(refSimDir +'t_start.npy')
	t_duration 	= np.load(refSimDir +'t_duration.npy')
	fluidVelMag = np.load(refSimDir +'fluidVelMag.npy')
	fluidSwitch = np.load(refSimDir +'fluidSwitch.npy')

	print(t_start)
	print(fluidSwitch)

	# modify simulation parameters here to compare
	args.visc = 0.001

# id for symmetric layout
def check_inflow_timestamp(i, j, id, t):
	swithOn = False
	if id == 1:
		start 	= t_start[i][j]
		end   	= start + t_duration[i][j]
		swithOn = fluidSwitch[i][j]
	elif id == 2:
		start 	= t_start[i+n_x][j]
		end	  	= start + t_duration[i+n_x][j]
		swithOn = fluidSwitch[i+n_x][j]
	elif id == 3:
		start 	= t_start[i][j+n_z]
		end	  	= start + t_duration[i][j+n_z]
		swithOn = fluidSwitch[i][j+n_z]
	elif id == 4:
		start 	= t_start[i+n_x][j+n_z]
		end	  	= start + t_duration[i+n_x][j+n_z]
		swithOn = fluidSwitch[i+n_x][j+n_z]

	if swithOn and t >= start and t<= end:
		# print('True, (i,j,id,t) = ({:d},{:d},{:d},{:d})'.format(i,j,id,t))
		return True
	else:
		# print('False, (i,j,id,t) = ({:d},{:d},{:d},{:d})'.format(i,j,id,t))
		return False

def update_inflow_obstacle(t):
	global phiFluidIn
	phiFluidIn.setConst(999)
	update = False
	for i in range(n_x):
		for j in range(n_z):
			for ii in range(nn_x):
				for jj in range(nn_z):
					p_1 = p_center + vec3(i*d_x+ii*dd_x, 0, j*d_z+jj*dd_z)
					p_2 = p_center + vec3(-i*d_x+ii*dd_x, 0, j*d_z+jj*dd_z)
					p_3 = p_center + vec3(i*d_x+ii*dd_x, 0, -j*d_z+jj*dd_z)
					p_4 = p_center + vec3(-i*d_x+ii*dd_x, 0, -j*d_z+jj*dd_z)
					n_1 = vec3(math.cos(altitude+d_alt[i][j])*math.cos(azimuth+d_azi[i][j]), math.sin(altitude+d_alt[i][j]), math.cos(altitude+d_alt[i][j])*math.sin(azimuth+d_azi[i][j]))
					n_2 = vec3(math.cos(altitude+d_alt[i+n_x][j])*math.cos(azimuth+d_azi[i+n_x][j]), math.sin(altitude+d_alt[i+n_x][j]), math.cos(altitude+d_alt[i+n_x][j])*math.sin(azimuth+d_azi[i+n_x][j]))
					n_3 = vec3(math.cos(altitude+d_alt[i][j+n_z])*math.cos(azimuth++d_azi[i][j+n_z]), math.sin(altitude+d_alt[i][j+n_z]), math.cos(altitude+d_alt[i][j+n_z])*math.sin(azimuth+d_azi[i][j+n_z]))
					n_4 = vec3(math.cos(altitude+d_alt[i+n_x][j+n_z])*math.cos(azimuth+d_azi[i+n_x][j+n_z]), math.sin(altitude+d_alt[i+n_x][j+n_z]), math.cos(altitude+d_alt[i+n_x][j+n_z])*math.sin(azimuth+d_azi[i+n_x][j+n_z]))

					if check_inflow_timestamp(i,j,1,t):
						phiFluidIn.join(Cylinder(parent=s, center=gs*p_1, radius=radius, z=n_1).computeLevelset())
					if check_inflow_timestamp(i,j,2,t):
						phiFluidIn.join(Cylinder(parent=s, center=gs*p_2, radius=radius, z=n_2).computeLevelset())
					if check_inflow_timestamp(i,j,3,t):
						phiFluidIn.join(Cylinder(parent=s, center=gs*p_3, radius=radius, z=n_3).computeLevelset())
					if check_inflow_timestamp(i,j,4,t):
						phiFluidIn.join(Cylinder(parent=s, center=gs*p_4, radius=radius, z=n_4).computeLevelset())

					if check_inflow_timestamp(i,j,1,t):
						fluidVel 	= Cylinder(parent=s, center=gs*p_1, radius=radius, z=n_1)
						fluidSetVel = fluidVelMag[i][j]*n_1
						fluidVel.applyToGrid( grid=vel , value=fluidSetVel )
					if check_inflow_timestamp(i,j,2,t):
						fluidVel 	= Cylinder(parent=s, center=gs*p_2, radius=radius, z=n_2)
						fluidSetVel = fluidVelMag[i+n_x][j]*n_2
						fluidVel.applyToGrid( grid=vel , value=fluidSetVel )	
					if check_inflow_timestamp(i,j,3,t):
						fluidVel 	= Cylinder(parent=s, center=gs*p_3, radius=radius, z=n_3)
						fluidSetVel = fluidVelMag[i][j+n_z]*n_3
						fluidVel.applyToGrid( grid=vel , value=fluidSetVel )	
					if check_inflow_timestamp(i,j,4,t):
						fluidVel 	= Cylinder(parent=s, center=gs*p_4, radius=radius, z=n_4)
						fluidSetVel = fluidVelMag[i+n_x][j+n_z]*n_4
						fluidVel.applyToGrid( grid=vel , value=fluidSetVel )	

					if check_inflow_timestamp(i,j,1,t) or check_inflow_timestamp(i,j,2,t) or check_inflow_timestamp(i,j,3,t) or check_inflow_timestamp(i,j,4,t):
						mapGridToPartsVec3(source=vel, parts=pp, target=pVel )
						update = True
	
	global flags, phi 					
	if update:
		# Reset flags, phiObs and phi (Because of moving objects we have to reset flags for every step)
		flags.initDomain(boundaryWidth=bWidth, phiWalls=phiObs)
		if doOpen:
			setOpenBound(flags, bWidth,'xXyYzZ',FlagOutflow|FlagEmpty) 
		
		phi.join(phiFluidIn)	

		# Sample new particles, but only into parts of phiFluidIn that do not contain any particles anymore
		sampleLevelsetWithParticles(phi=phiFluidIn, flags=flags, parts=pp, discretization=particleNumber, randomness=randomness, refillEmpty=True)

		# Set flow cells to fluid (1), and other non-obstacle cells to empty (4)
		flags.updateFromLevelset(phi) # Important! This has to be called after the sampling. Otherwise the 'refillEmpty' option does not work (function would break too early out of sampling loop)

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

def save_data(t):
	phi.createMesh(mesh_fluid)	 # note: these meshes are created by fast marching only, should smooth geometry and normals before rendering
	if (saveData):
		vel.save(simDir + 'flipVel_%04d.uni' % t)
		pp.save( simDir + 'flipParts_%04d.uni' % t)
		phi.save(simDir + 'flipLevelSet_%04d.uni' % t)
		mesh_fluid.save(simDir + 'flip_%04d.gz' % t)
		gui.screenshot( simDir + 'flip_%04d.png' % t)	

# main program
# load_scene_config()
save_scene_config()

gui = Gui()
gui.show()
gui.setCamPos(0., 0., -2)
gui.setCamRot(0,0,0)

for t in range(steps):
	maxVel = vel.getMax()
	s.adaptTimestep( maxVel )
	mantaMsg('\nFrame %i, timestep %f, timeTotal %f' % (s.frame, s.timestep, s.timeTotal))

	update_inflow_obstacle(t)
	flip(t)
	s.step()
	save_data(t)