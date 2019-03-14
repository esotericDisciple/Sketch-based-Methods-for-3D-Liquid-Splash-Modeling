#ifndef __CGAL_WRAPPER_H__
#define __CGAL_WRAPPER_H__

#define CGAL_EIGEN3_ENABLED

#include <CGAL/trace.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Implicit_surface_3.h>

#include <CGAL/Point_with_normal_3.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/jet_estimate_normals.h>
#include <CGAL/vcm_estimate_normals.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/property_map.h>
 
#include <CGAL/compute_average_spacing.h>
#include <CGAL/Polygon_mesh_processing/distance.h>

#include <CGAL/Poisson_reconstruction_function.h>
#include <CGAL/poisson_surface_reconstruction.h>

#include <CGAL/IO/facets_in_complex_2_to_triangle_mesh.h>
#include <CGAL/IO/read_xyz_points.h>


// Types
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::FT FT;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
typedef Kernel::Sphere_3 Sphere;

typedef CGAL::Polyhedron_3<Kernel> Polyhedron;
typedef CGAL::Point_with_normal_3<Kernel> Point_with_normal;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron;
typedef CGAL::Poisson_reconstruction_function<Kernel> Poisson_reconstruction_function;
typedef CGAL::Surface_mesh_default_triangulation_3 STr;
typedef CGAL::Surface_mesh_complex_2_in_triangulation_3<STr> C2t3;
typedef CGAL::Implicit_surface_3<Kernel, Poisson_reconstruction_function> Surface_3;

typedef std::pair<Point, Vector> Pwn;
typedef std::pair<Point, Vector> PointVectorPair;
typedef std::vector<Point_with_normal> PointNormalList;
typedef std::vector<PointVectorPair> PointVectorList;

// Concurrency
#ifdef CGAL_LINKED_WITH_TBB
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif

using namespace std;

//--- forward declaration
void run_pca_estimate_normals(PointVectorList& points, // input points + output normals
	unsigned int nb_neighbors_pca_normals);

void run_jet_estimate_normals(PointVectorList& points, // input points + output normals
	unsigned int nb_neighbors_jet_fitting_normals);

void run_vcm_estimate_normals(PointVectorList &points, // input points + output normals
	double R, // radius of the offset
	double r);

void run_mst_orient_normals(PointVectorList& points, // input points + input/output normals
	unsigned int nb_neighbors_mst);


//--- wrap functions
void CGAL_Poisson_Reconstruction_Function(const string &in_dir, const string &out_dir, const string &filename)
{
	std::string in_filename = in_dir + "\\" + filename + ".xyz";

	std::vector<Pwn> points;
	std::ifstream stream(in_filename);
	if (!stream ||
		!CGAL::read_xyz_points(
			stream,
			std::back_inserter(points),
			CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Pwn>()).
			normal_map(CGAL::Second_of_pair_property_map<Pwn>())))
	{
		printf("ERROR: cannot open %s\n", in_filename); getchar();
	}


	// Poisson options
	FT sm_angle = 20.0; // Min triangle angle in degrees.
	FT sm_radius = 30;	// Max triangle size w.r.t. point set average spacing.
	FT sm_distance = 0.375; // Surface Approximation error w.r.t. point set average spacing.

	char out_filename_c[1024];
	sprintf(out_filename_c, string(out_dir + "\\" + filename + "-%d-%d-%.3f.off").c_str(), int(sm_angle), int(sm_radius), float(sm_distance));
	std::string out_filename = string(out_filename_c);

	Polyhedron output_mesh;

	double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>
		(points, 6, CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Pwn>()));

	if (CGAL::poisson_surface_reconstruction_delaunay
	(points.begin(), points.end(),
		CGAL::First_of_pair_property_map<Pwn>(),
		CGAL::Second_of_pair_property_map<Pwn>(),
		output_mesh, average_spacing, sm_angle, sm_radius, sm_distance))
	{
		std::ofstream out(out_filename);
		out << output_mesh;
	}

}

void CGAL_Poisson_Reconstruction_Class(const string &in_dir, const string &out_dir, const string &filename)
{
	std::string in_filename = in_dir + "\\" + filename + ".xyz";

	// Poisson options
	FT sm_angle = 20.0; // Min triangle angle in degrees.
	FT sm_radius = 30; // Max triangle size w.r.t. point set average spacing.
	FT sm_distance = 0.375; // Surface Approximation error w.r.t. point set average spacing.

							// Reads the point set file in points[].
							// Note: read_xyz_points_and_normals() requires an iterator over points
							// + property maps to access each point's position and normal.
							// The position property map can be omitted here as we use iterators over Point_3 elements.
	PointNormalList points;
	std::ifstream stream(in_filename);
	if (!stream ||
		!CGAL::read_xyz_points(
			stream,
			std::back_inserter(points),
			CGAL::parameters::normal_map(CGAL::make_normal_of_point_with_normal_map(PointNormalList::value_type()))))
	{
		printf("ERROR: cannot open %s\n", in_filename); getchar();
	}

	// Creates implicit function from the read points using the default solver.

	// Note: this method requires an iterator over points
	// + property maps to access each point's position and normal.
	// The position property map can be omitted here as we use iterators over Point_3 elements.
	Poisson_reconstruction_function function(points.begin(), points.end(),
		CGAL::make_normal_of_point_with_normal_map(PointNormalList::value_type()));

	// Computes the Poisson indicator function f()
	// at each vertex of the triangulation.
	if (!function.compute_implicit_function())
	{
		printf("ERROR: compute_implicit_function\n"); getchar();
	}

	// Computes average spacing
	FT average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(points, 6 /* knn = 1 ring */);

	// Gets one point inside the implicit surface
	// and computes implicit function bounding sphere radius.
	Point inner_point = function.get_inner_point();
	Sphere bsphere = function.bounding_sphere();
	FT radius = std::sqrt(bsphere.squared_radius());

	// Defines the implicit surface: requires defining a
	// conservative bounding sphere centered at inner point.
	FT sm_sphere_radius = 5.0 * radius;
	FT sm_dichotomy_error = sm_distance*average_spacing / 1000.0; // Dichotomy error must be << sm_distance
	Surface_3 surface(function,
		Sphere(inner_point, sm_sphere_radius*sm_sphere_radius),
		sm_dichotomy_error / sm_sphere_radius);

	// Defines surface mesh generation criteria
	CGAL::Surface_mesh_default_criteria_3<STr> criteria(sm_angle,  // Min triangle angle (degrees)
		sm_radius*average_spacing,  // Max triangle size
		sm_distance*average_spacing); // Approximation error

									  // Generates surface mesh with manifold option
	STr tr; // 3D Delaunay triangulation for surface mesh generation
	C2t3 c2t3(tr); // 2D complex in 3D Delaunay triangulation
	CGAL::make_surface_mesh(c2t3,                                 // reconstructed mesh
		surface,                              // implicit surface
		criteria,                             // meshing criteria
		CGAL::Manifold_with_boundary_tag());  // require manifold mesh

	if (tr.number_of_vertices() == 0)
	{
		printf("ERROR: number_of_vertices\n"); getchar();
	}

	char out_filename_c[1024];
	sprintf(out_filename_c, string(out_dir + "\\" + filename + "-%d-%d-%.3f.off").c_str(), int(sm_angle), int(sm_radius), float(sm_distance));
	std::string out_filename = string(out_filename_c);

	// saves reconstructed surface mesh
	std::ofstream out(out_filename);
	Polyhedron output_mesh;
	CGAL::facets_in_complex_2_to_triangle_mesh(c2t3, output_mesh);
	out << output_mesh;
}

std::vector<PointVectorPair>  CGAL_Run_Estimate_Normal(const string &in_filename)
{
	PointVectorList orientedPoints;

 	// Reads a .xyz point set file in points[].
	std::ifstream stream(in_filename);
	if (!stream ||
		!CGAL::read_xyz_points(stream,
			std::back_inserter(orientedPoints),
			CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())))
	{
 		printf("ERROR: cannot open %s\n", in_filename); getchar();
	}


	// Normals Computing options
	unsigned int nb_neighbors_pca_normals = 18; // K-nearest neighbors = 3 rings (estimate normals by PCA)
	unsigned int nb_neighbors_jet_fitting_normals = 18; // K-nearest neighbors (estimate normals by Jet Fitting)
	unsigned int nb_neighbors_mst = 18; // K-nearest neighbors (orient normals by MST)
	double offset_radius_vcm = 0.1; // Offset radius (estimate normals by VCM)
	double convolve_radius_vcm = 0; // Convolve radius (estimate normals by VCM)
	std::string estimate = "quadric"; // estimate normals by jet fitting
	//std::string estimate = "plane"; // estimate normals by  linear least squares fitting of a plane
	//std::string estimate = "vcm"; // estimate normals by  using the Voronoi Covariance Measure of the point set.
	std::string orient = "MST"; // orient normals using a Minimum Spanning Tree

	 
	//***************************************
    // Computes normals
    //***************************************
    // Estimates normals direction.
    if (estimate == "plane")
      run_pca_estimate_normals(orientedPoints, nb_neighbors_pca_normals);
    else if (estimate == "quadric")
      run_jet_estimate_normals(orientedPoints, nb_neighbors_jet_fitting_normals);
    else if (estimate == "vcm")
      run_vcm_estimate_normals(orientedPoints, offset_radius_vcm, convolve_radius_vcm);

	// Orient normals.
	if (orient == "MST")
		run_mst_orient_normals(orientedPoints, nb_neighbors_mst);

	return orientedPoints;
}


//--- Private functions

// Computes normals direction by Principal Component Analysis
void run_pca_estimate_normals(PointVectorList& points, // input points + output normals
	unsigned int nb_neighbors_pca_normals) // number of neighbors
{
	CGAL::Timer task_timer; task_timer.start();
	//std::cerr << "Estimates Normals Direction by PCA (k=" << nb_neighbors_pca_normals << ")...\n";

	// Estimates normals direction.
	// Note: pca_estimate_normals() requires an iterator over points
	// as well as property maps to access each point's position and normal.
	CGAL::pca_estimate_normals<Concurrency_tag>(points,
		nb_neighbors_pca_normals,
		CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>()).
		normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));


	//std::size_t memory = CGAL::Memory_sizer().virtual_size();
	//std::cerr << "done: " << task_timer.time() << " seconds, " << (memory >> 20) << " Mb allocated" << std::endl;
}

// Computes normals direction by Jet Fitting
void run_jet_estimate_normals(PointVectorList& points, // input points + output normals
	unsigned int nb_neighbors_jet_fitting_normals) // number of neighbors
{
	CGAL::Timer task_timer; task_timer.start();
	//std::cerr << "Estimates Normals Direction by Jet Fitting (k=" << nb_neighbors_jet_fitting_normals << ")...\n";

	// Estimates normals direction.
	// Note: jet_estimate_normals() requires an iterator over points
	// + property maps to access each point's position and normal.
	CGAL::jet_estimate_normals<Concurrency_tag>
		(points,
			nb_neighbors_jet_fitting_normals,
			CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>()).
			normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));


	//std::size_t memory = CGAL::Memory_sizer().virtual_size();
	//std::cerr << "done: " << task_timer.time() << " seconds, " << (memory >> 20) << " Mb allocated" << std::endl;
}

// Compute normals direction using the VCM
void run_vcm_estimate_normals(PointVectorList &points, // input points + output normals
	double R, // radius of the offset
	double r) { // radius used during the convolution
	CGAL::Timer task_timer; task_timer.start();
	//std::cerr << "Estimates Normals Direction using VCM (R=" << R << " and r=" << r << ")...\n";

	// Estimates normals direction.
	// Note: vcm_estimate_normals() requires an iterator over points
	// + property maps to access each point's position and normal.
	CGAL::vcm_estimate_normals(points, R, r,
		CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>()).
		normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));

	//std::size_t memory = CGAL::Memory_sizer().virtual_size();
	//std::cerr << "done: " << task_timer.time() << " seconds, " << (memory >> 20) << " Mb allocated" << std::endl;
}

// Hoppe92 normal orientation using a Minimum Spanning Tree.
void run_mst_orient_normals(PointVectorList& points, // input points + input/output normals
	unsigned int nb_neighbors_mst) // number of neighbors
{
	//std::cerr << "Orients Normals with a Minimum Spanning Tree (k=" << nb_neighbors_mst << ")...\n";
	CGAL::Timer task_timer; task_timer.start();

	// Orients normals.
	// Note: mst_orient_normals() requires an iterator over points
	// as well as property maps to access each point's position and normal.
	PointVectorList::iterator unoriented_points_begin =
		CGAL::mst_orient_normals(points,
			nb_neighbors_mst,
			CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>()).
			normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));

	// Optional: delete points with an unoriented normal
	// if you plan to call a reconstruction algorithm that expects oriented normals.
	points.erase(unoriented_points_begin, points.end());

	//std::size_t memory = CGAL::Memory_sizer().virtual_size();
	//std::cerr << "done: " << task_timer.time() << " seconds, " << (memory >> 20) << " Mb allocated" << std::endl;
}


#endif