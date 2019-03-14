#ifndef __SPSR_WRAPPER_H__
#define __SPSR_WRAPPER_H__

#include <Windows.h>
#include<string>

using namespace std;

void Adaptive_Multigrid_Solvers_Surface_Reconstruction(const string in_dir, const string out_dir, const string filename_base)
{
	string in_filename					= in_dir + "\\" + filename_base + ".orientedPoints.ply";
	string out_filename_PoissonRecon	= out_dir + "\\" + filename_base + ".PoissonRecon.ply";
	string out_filename_SSDRecon		= out_dir + "\\" + filename_base + ".SSDRecon.ply";

	string cmd_msg;

	// Reconstructs a triangle mesh from a set of oriented 3D points by solving a Poisson system 
	// (solving a 3D Laplacian system with positional value constraints
	cmd_msg = "externallibs\\AdaptiveSolvers.x64\\PoissonRecon.exe --in " + in_filename + " --out " + out_filename_PoissonRecon + " --depth 10";
	system(cmd_msg.c_str());

	// Reconstructs a surface mesh from a set of oriented 3D points by solving for a Smooth Signed Distance function 
	// (solving a 3D bi-Laplacian system with positional value and gradient constraints) 
	cmd_msg = "externallibs\\AdaptiveSolvers.x64\\SSDRecon.exe --in " + in_filename + " --out " + out_filename_SSDRecon + " --depth 10";
	system(cmd_msg.c_str());
}

#endif