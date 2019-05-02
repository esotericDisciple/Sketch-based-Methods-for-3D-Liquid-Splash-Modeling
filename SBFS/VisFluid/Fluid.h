#ifndef __EXAMPLE_H__
#define __EXAMPLE_H__

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <assert.h>

#include "zlib.h"
#include "Tri_Mesh.h"

#ifdef USE_CGAL
#include "CGAL_Wrapper.h"
#include "tinyply_Wrapper.h"
#endif

#include "SPSR_Wrapper.h"
#include "Manta_Wrapper.h"

// ---  MACROS
#define PRINTF					printf
#define ERROR_CHECK(error, msg) { if(error) {printf(msg); printf("check line number %d\n", __LINE__); getchar();} }
#define CLAMP(a, l, h)			( ((a)>(h))?(h):(((a)<(l))?(l):(a)) )
#define INDEX(i,j,k)			( (int)(i) +grid_res*(int)(j) + grid_res*grid_res*(int)(k) ) // assume the grid is uniform
#define FOR_EVERY_CELL		\
	for(int k=0, id=0; k<grid_res; k++)	\
	for(int j=0; j<grid_res; j++)			\
	for(int i=0; i<grid_res; i++, id++)

using namespace std;

template <class TYPE>
class Fluid :public Tri_Mesh<TYPE>
{
public:

	// scene settings 
	bool extend;            // 4 channel version or single channel version
	string scene_name;
	int cur_frame;
	int batch_size, num_batch, iepoch;			// batch size used in training/visualization
	TYPE threshold_value;   // value used for surface reconstruction from binary levelset
	TYPE fps;
	int max_frames, max_particles, max_vertices;
	int max_streamline_length, max_streamline_number;
	int start_frame, end_frame;				// start_frame and end_frame, inclusive for the animation
	int draw_start_frame, draw_end_frame;	// rendering range

	// pathes
	string in_manta_basePath, in_tensorflow_path, in_matlab_basePath;

	// particle data here
	int num_pathlines;
	// pathline_animation[i]: all particle data in i-th frame(manta frame starts from index one)
	Tri_Mesh<TYPE> **pathline_animation;			
	// pathlines[i]: the trajectory of i-th particle
	TYPE**	pathline_particle;
	TYPE*	pathcolors;

	// grid data here 
	int		grid_res;
	TYPE*	grid_vel;  // grid data is large, so only load one frame everytime, used for format conversion
	TYPE*	grid_phi;
	TYPE**	grid_levelset; // scale data
	TYPE**	grid_sketch;   // scale data
	TYPE**	grid_levelset_vel;
	TYPE**	grid_sketch_vel;

	// num_streamlines[i]: number of streamlines at i-th frame
	int* num_streamlines;
	// streamline_length[t][i]: length of i-th streamline at t-th frame 
	int** streamline_length;
	// streamline_grid[t][i]: data of i-th streamline at t-th frame 
	TYPE*** streamline_grid;
	TYPE*	streamline_colors;
	int**   streamline_labels;

	// mesh data here: note that manta normalize mesh to unit cube around 0 when exporting, pos -= toVec3f(gs)*0.5; pos *= dx;
	Tri_Mesh<TYPE> **lvst_mesh_animation;

	// validate result from tensorflow
	Tri_Mesh<TYPE> **train_animation;
	Tri_Mesh<TYPE> **predict_animation;

	vector<string> addrs_streamline;
	vector<string> addrs_sketch;
	vector<string> addrs_lvst;

public:
	Fluid()
	{
		streamline_grid			= NULL;
		streamline_colors		= NULL;
		streamline_labels		= NULL;
		grid_vel				= NULL;
		pathline_animation		= NULL;
		lvst_mesh_animation		= NULL;

		grid_sketch				= NULL;
		grid_levelset			= NULL;

		grid_levelset_vel		= NULL;
		grid_sketch_vel			= NULL;

		train_animation			= NULL;
		predict_animation		= NULL;

		start_frame				= 1;
		end_frame				= 1;
		num_pathlines			= 0;
		cur_frame				= 1;
		fps						= 15;
		grid_res				= 128;
		draw_start_frame		= start_frame;
		draw_end_frame			= end_frame;
		max_frames				= 120;
		max_particles			= 5000000;

		max_streamline_length	= 500; // oringinal data need have extream long/short streamlines
		max_streamline_number	= 500;

		max_vertices			= 1000000;

	}

	~Fluid() {}

	void Initialize()
	{
		//-- parameters
		start_frame		= 20;
		end_frame		= 99;
		grid_res		= 128;
		extend			= true;
		batch_size		= 4;
		num_batch		= 5;
		threshold_value = 0.5;
		iepoch			= 16;

		in_manta_basePath		= "..\\manta\\sbfs_scenes\\data";
		in_tensorflow_path		= extend? "..\\tensorflow\\data_extend_REALGAN_cluster\\result" : "..\\tensorflow\\data\\result";
		in_matlab_basePath		= "..\\VisMat\\data\\4096";
		
		const string out_vs_basePath		= "data";
		const string sceneName				= "sbfs_flip_water_pipes_10000";
		const string parts_filename			= "flipParts";
		const string mesh_filename			= "flip";
		const string vel_filename			= "flipVel";
		const string levelset_filename		= "flipLevelSet";
		const string streamline_filename	= "flipStreamline";
		const string sketch_filename		= "flipSketchGrid";
		const string suffix					= ".uni";
		const int num_clusters				= 32;

		// -- functions
		bool fileCompressed					= true;
		bool process_meshes					= false;
		bool process_streamlines			= false;
		bool process_sketches				= false;

		bool run_visualization				= true;
		bool compare_result					= true;

		bool process_pathlines				= false;
		bool exportOrientedPoints			= false;
		bool process_surface_reconstruction = false;
		bool process_grid_conversion		= false;

		Set_Scene_Name(sceneName);

		if (run_visualization)
		{
			 Surface_Reconstruction_From_Levelset();
		}
		if (compare_result)
		{
			//Load_Sketch_Levelset_Grid_Uni(out_vs_basePath);
			
			Load_Tensorflow_Prediction_Batch(in_tensorflow_path, extend);

			//Load_Tensorflow_Animation();
			//Load_Tensorflow_Demo_Sequence();
			//Load_Tensorflow_Demo_Interpolate();
		}
		if (process_pathlines) //-- load and build manta particles pathlines
		{
			/*Load_Flip_Parts_TXT(in_manta_basePath, sceneName, parts_filename);
			Build_PathLines();
			Export_Pathlines(out_vs_basePath, sceneName);*/

			Import_Pathlines(out_vs_basePath, sceneName, false, num_clusters);
		}	
		if (process_streamlines) //-- load streamlines generated by MATLAB
		{
			Load_Streamline_Grid_TXT(in_matlab_basePath, sceneName, streamline_filename, true);
		}
		if (process_sketches)
		{
			Load_Sketch_Grid_Bin(in_matlab_basePath, sceneName, sketch_filename); //-- load sketch grid generated by MATLAB
		}
		if (process_grid_conversion) //-- convert raw/uni data from manta to matlab binary
		{
			Batch_Convert_Grid_Bin(grid_vel, grid_res, grid_res, grid_res, true, in_manta_basePath, out_vs_basePath, sceneName, vel_filename, suffix);
			Batch_Convert_Grid_Bin(grid_phi, grid_res, grid_res, grid_res, false, in_manta_basePath, out_vs_basePath, sceneName, levelset_filename, suffix);
		}

		if (process_meshes) //-- load mesh generated by manta
		{
			Load_Manta_Meshes(in_manta_basePath, out_vs_basePath, sceneName, mesh_filename, fileCompressed, exportOrientedPoints);
			//Load_Tensorflow_Meshes(in_tensorflow_path);
		}
		if (process_surface_reconstruction)
		{
#ifdef USE_CGAL
			//Oriented_Points_Surface_Reconstruction(out_vs_basePath, out_vs_basePath, sceneName, mesh_filename);
			Load_Reconstruction_Meshes(out_vs_basePath, sceneName, mesh_filename);
#endif
		}
	}

	// update current frame, rendering frame range etc..
	void Update(int iterations, int type = 2)
	{
		// for streamlines
		cur_frame = iterations;
		//cur_frame = CLAMP(iterations, start_frame, end_frame);
		
		// for pathlines
		if(type == 0 ) // show all frames
			iterations > end_frame ? draw_end_frame = end_frame : draw_end_frame = iterations;
		else if (type == 1 ) // loop animation
			draw_end_frame = iterations % (end_frame - start_frame + 1) + start_frame;
		else if (type == 2) // show last nframes
		{
			int nframe = 10;
			if (iterations < start_frame + nframe)
			{
				draw_start_frame = start_frame;
				draw_end_frame = iterations;
			}
			else if (iterations < end_frame)
			{
				draw_start_frame = iterations - nframe;
				draw_end_frame = iterations;
			}
			else
			{
				draw_start_frame = end_frame - nframe;
				draw_end_frame = end_frame;
			}
		}
		else
		{
			if (iterations > end_frame)
			{
				draw_start_frame++;
				if (draw_start_frame > draw_end_frame)
					draw_start_frame = draw_end_frame;

				draw_end_frame = end_frame;
			}
			else
			{
				draw_end_frame = iterations;
				draw_start_frame = start_frame;
			}
		}

 	}
	
	// call manta python program for levelset surface reconstruction
	void Surface_Reconstruction_From_Levelset()
	{
		string manta_dir = in_manta_basePath;
		string result_dir = in_tensorflow_path;
		string mat_dir = "..\\VisMat\\data";

		printf("start levelset surface reconstruction\n");
		char cmd_msg[1024];
		string msg = "manta sbfs_tf_visualize_sketch_extend_vs.py --batch_size %d --num_batch %d --iepoch %d --threshold_value %f --result_dir %s --mat_dir %s --manta_dir %s";
		sprintf(cmd_msg, msg.c_str(), batch_size, num_batch, iepoch, threshold_value, result_dir.c_str(), mat_dir.c_str(), manta_dir.c_str());
		printf("%s\n", cmd_msg);
		system(cmd_msg);
		printf("end levelset surface reconstruction");
	}

	void Surface_Reconstruction_From_Levelset_Update(int update_frame, float update_threshold)
	{
		string manta_dir = in_manta_basePath;
		string result_dir = in_tensorflow_path;
		string mat_dir = "..\\VisMat\\data";

		printf("start update levelset surface reconstruction, frame = %d, threshold_value = %f\n", update_frame, update_threshold);
		char cmd_msg[1024];
		string msg = "manta sbfs_tf_visualize_sketch_extend_vs_update.py --update_frame %d --batch_size %d --num_batch %d --iepoch %d --threshold_value %f --result_dir %s --mat_dir %s --manta_dir %s";
		sprintf(cmd_msg, msg.c_str(), update_frame, batch_size, num_batch, iepoch, update_threshold, result_dir.c_str(), mat_dir.c_str(), manta_dir.c_str());
		//printf("%s\n", cmd_msg);
		system(cmd_msg);
		printf("end update levelset surface reconstruction, frame = %d, threshold_value = %f\n", update_frame, update_threshold);


		// reload the updated frame
		// the addresses are already loaded.
		string os_sep = "\\";

		string sketch_addr	= addrs_sketch[update_frame];
		string lvst_addr	= addrs_lvst[update_frame];

		string addr_lvst = lvst_addr;
		string filename_lvst = addr_lvst.substr(addr_lvst.rfind(os_sep) + 1, addr_lvst.rfind('.') - addr_lvst.rfind(os_sep) - 1);
		string frame_num = filename_lvst.substr(filename_lvst.rfind('_') + 1);
		string addr_lvst_pre = addr_lvst.substr(0, addr_lvst.rfind(os_sep));
		string addr_lvst_pre_pre = addr_lvst_pre.substr(0, addr_lvst_pre.rfind(os_sep));
		string simname = addr_lvst_pre.substr(addr_lvst_pre.rfind(os_sep) + 1);

		string addr_sketch = sketch_addr;
		string addr_sketch_pre = addr_sketch.substr(0, addr_sketch.rfind(os_sep));
		string addr_sketch_pre_pre = addr_sketch_pre.substr(0, addr_sketch_pre.rfind(os_sep));
		string seed_num = addr_sketch_pre_pre.substr(addr_sketch_pre_pre.rfind(os_sep) + 1);
		//std::cout << seed_num << simname << filename_lvst << frame_num;

		string path_pred_mesh = result_dir + os_sep + seed_num + "#" + simname + "#" + filename_lvst + "_pred.obj";

		// load mesh
		predict_animation[update_frame]->Read_Manta_Mesh_OBJ(path_pred_mesh.c_str());
		predict_animation[update_frame]->Build_Connectivity();
		predict_animation[update_frame]->Lap_Smooth_Mesh();
		predict_animation[update_frame]->Build_VN(predict_animation[update_frame]->X);
	}

	// load data addresses
	void Load_Data_Addrs_Comp(const string file_addrs)
	{
		string filename = string("..\\tensorflow\\data\\result\\") + file_addrs;
		fstream file(filename, std::fstream::in);
		ERROR_CHECK(!file.is_open(), ("ERROR: cannot open file: %s\n", filename.c_str()));
		string line;
		while (getline(file, line, '\n'))
		{
			addrs_sketch.push_back(line.substr(0, line.find(' ')));
			addrs_lvst.push_back(line.substr(line.find(' ') + 1));

			/*std::cout << addrs_sketch.back() << std::endl;
			std::cout << addrs_lvst.back() << std::endl;
			getchar();*/
		}
	}

	void Load_Data_Addrs(const string in_tensorflow_path, const string file_addrs, bool extend)
	{
		vector<string> addrs_vel;
		string filename = in_tensorflow_path + "\\" + file_addrs;
		fstream file(filename, std::fstream::in);
		ERROR_CHECK(!file.is_open(), ("ERROR: cannot open file: %s\n", filename.c_str()));
		string line;
		while (getline(file, line, '\n'))
		{
			if (!extend)
			{
				addrs_sketch.push_back(line.substr(0, line.find(' ')));
				addrs_lvst.push_back(line.substr(line.find(' ') + 1));

				/*std::cout << addrs_sketch.back() << std::endl;
				std::cout << addrs_lvst.back() << std::endl;
				getchar();*/
			} else{
				addrs_sketch.push_back(line.substr(0, line.find(' ')));
				string subline = line.substr(line.find(' ') + 1);
				addrs_lvst.push_back(subline.substr(0, subline.find(' ')));
				addrs_vel.push_back(subline.substr(subline.find(' ') + 1));

				/*std::cout << addrs_sketch.back() << std::endl;
				std::cout << addrs_lvst.back() << std::endl;
				std::cout << addrs_vel.back() << std::endl;
				getchar();*/
			}
		}
	}

	void Get_Data_Addrs_Next_Batch(int ibatch, int batch_size, vector<string> &data_x, vector<string> &data_y)
	{
		data_x.clear();
		data_y.clear();
		for (int i = ibatch*batch_size; i < ibatch*batch_size + batch_size; i++)
		{
			data_x.push_back(addrs_sketch[i]);
			data_y.push_back(addrs_lvst[i]);
		}
	}

	// load animation sequence generated from predicted fluid occupancy grid and velocity grid
	void Load_Tensorflow_Animation()
	{
		string inpath = "..\\tensorflow\\demo";
		int prediction_frame = 15; // the prediction frame 
		start_frame = 0;
		end_frame = 99;
		Set_Scene_Name("animation_" + to_string(prediction_frame));

		int batch_size = 4;
		int num_batch = 20;
		string filetype = "test";
		string addr_filename;
		if (filetype == "test")
			addr_filename = "test_addrs.txt";
		else
			addr_filename = "train_addrs.txt";
		int current_frame = 0;
		string os_sep = "\\";
		Load_Data_Addrs_Comp(addr_filename);

		// allocate memory for streamlines
		if (streamline_grid == NULL)
		{
			num_streamlines = new int[max_frames];
			streamline_length = new int*[max_frames];
			streamline_grid = new TYPE**[max_frames];
			for (int f = 0; f < max_frames; f++)
			{
				streamline_length[f] = new int[max_streamline_number];
				streamline_grid[f] = new TYPE*[max_streamline_number];
				for (int n = 0; n < max_streamline_number; n++)
					streamline_grid[f][n] = new TYPE[max_streamline_length * 3];
			}
			streamline_colors = new TYPE[3 * max_streamline_number];
		}
		float center = 0.5 * grid_res;
		float scale = 1.0 / grid_res;
		// allocate memory for meshes
		if (predict_animation == NULL)
		{
			predict_animation = new Tri_Mesh<TYPE>*[max_frames];
			for (int i = 0; i < max_frames; i++)
			{
				predict_animation[i] = new Tri_Mesh<TYPE>(max_vertices);
			}
		}

		string addr_lvst = addrs_lvst[prediction_frame];
		string filename_lvst = addr_lvst.substr(addr_lvst.rfind(os_sep) + 1, addr_lvst.rfind('.') - addr_lvst.rfind(os_sep) - 1);
		string frame_num = filename_lvst.substr(filename_lvst.rfind('_') + 1);
		string addr_lvst_pre = addr_lvst.substr(0, addr_lvst.rfind(os_sep));
		string addr_lvst_pre_pre = addr_lvst_pre.substr(0, addr_lvst_pre.rfind(os_sep));
		string simname = addr_lvst_pre.substr(addr_lvst_pre.rfind(os_sep) + 1);

		string addr_sketch = addrs_sketch[prediction_frame];
		string addr_sketch_pre = addr_sketch.substr(0, addr_sketch.rfind(os_sep));
		string addr_sketch_pre_pre = addr_sketch_pre.substr(0, addr_sketch_pre.rfind(os_sep));
		string seed_num = addr_sketch_pre_pre.substr(addr_sketch_pre_pre.rfind(os_sep) + 1);

		printf("start loading tensorflow animation sequences\n");
		for (int current_frame = start_frame; current_frame <= end_frame; current_frame++)
		{
			// load streamline (all use the prediction frame)
			char path_streamline[1024];
			sprintf(path_streamline, string(inpath + "\\%04d\\" + seed_num + "#" + simname + "#" + "flipStreamline_" + frame_num + "_resampled.txt").c_str(), prediction_frame);
			fstream file(path_streamline, std::fstream::in);
			ERROR_CHECK(!file.is_open(), ("ERROR: cannot open file: %s\n", path_streamline));
			string line;
			int line_id = 0;
			while (getline(file, line, '\n'))
			{
				stringstream ssline(line);
				string token;
				int point_id = 0;
				while (getline(ssline, token, ' '))
					streamline_grid[current_frame][line_id][point_id++] = (atof(token.c_str()) - center) * scale;
				ERROR_CHECK(point_id % 3 != 0, "ERROR: streamline points not correct!\n");
				streamline_length[current_frame][line_id] = point_id / 3;
				line_id++;
				ERROR_CHECK(line_id >= max_streamline_number, "ERROR: too many streamlines!\n");
			}
			num_streamlines[current_frame] = line_id;
			file.close();

			char in_filename[1024];
			sprintf(in_filename, string(inpath +  "\\%04d\\" + "flip_%04d.gz").c_str(), prediction_frame, current_frame);
			//printf("%s\n", in_filename);
			predict_animation[current_frame]->Read_Manta_Mesh_GZ(in_filename);
			predict_animation[current_frame]->Build_Connectivity();
			//predict_animation[current_frame]->Lap_Smooth_Mesh();
			predict_animation[current_frame]->Build_VN(predict_animation[current_frame]->X);
			//printf("%d\n", predict_animation[current_frame]->number);
		}
		printf("end loading tensorflow animation sequences\n");

		Update(start_frame);
	}

	// compare streamlines, groundtruth and the prediction(in batch generated from tensorflow at evaluation)
	void Load_Tensorflow_Prediction_Batch(const string in_tensorflow_path, bool extend)
	{
		start_frame = 0;
		end_frame = start_frame;

		// allocate memory for streamlines
		if (streamline_grid == NULL)
		{
			num_streamlines = new int[max_frames];
			streamline_length = new int*[max_frames];
			streamline_grid = new TYPE**[max_frames];
			for (int f = 0; f < max_frames; f++)
			{
				streamline_length[f] = new int[max_streamline_number];
				streamline_grid[f] = new TYPE*[max_streamline_number];
				for (int n = 0; n < max_streamline_number; n++)
					streamline_grid[f][n] = new TYPE[max_streamline_length * 3];
			}
			streamline_colors = new TYPE[3 * max_streamline_number];
		}
		float center = 0.5 * grid_res;
		float scale = 1.0 / grid_res;
		// allocate memory for meshes
		if (train_animation == NULL)
		{
			train_animation = new Tri_Mesh<TYPE>*[max_frames];
			predict_animation = new Tri_Mesh<TYPE>*[max_frames];
			for (int i = 0; i < max_frames; i++)
			{
				train_animation[i] = new Tri_Mesh<TYPE>(max_vertices);
				predict_animation[i] = new Tri_Mesh<TYPE>(max_vertices);
			}
		}

		Set_Scene_Name("REALGAN");
		string addr_filename = "test_addrs.txt";
		int current_frame = 0;
		string os_sep = "\\";
		Load_Data_Addrs("..\\tensorflow\\shared", addr_filename, extend);
		for (int ibatch = 0; ibatch < num_batch; ibatch++)
		{
			vector<string> sketch_addrs, lvst_addrs;
			Get_Data_Addrs_Next_Batch(ibatch, batch_size, sketch_addrs, lvst_addrs);
			for (int i = 0; i < batch_size; i++)
			{
				string addr_lvst = lvst_addrs[i];
				string filename_lvst = addr_lvst.substr(addr_lvst.rfind(os_sep) + 1, addr_lvst.rfind('.') - addr_lvst.rfind(os_sep) - 1);
				string frame_num = filename_lvst.substr(filename_lvst.rfind('_') + 1);
				string addr_lvst_pre = addr_lvst.substr(0, addr_lvst.rfind(os_sep));
				string addr_lvst_pre_pre = addr_lvst_pre.substr(0, addr_lvst_pre.rfind(os_sep));
				string simname = addr_lvst_pre.substr(addr_lvst_pre.rfind(os_sep) + 1);

				string addr_sketch = sketch_addrs[i];
				string addr_sketch_pre = addr_sketch.substr(0, addr_sketch.rfind(os_sep));
				string addr_sketch_pre_pre = addr_sketch_pre.substr(0, addr_sketch_pre.rfind(os_sep));
				string seed_num = addr_sketch_pre_pre.substr(addr_sketch_pre_pre.rfind(os_sep) + 1);
				//std::cout << seed_num << simname << filename_lvst << frame_num;

				string path_test_mesh = in_tensorflow_path + os_sep + seed_num + "#" + simname + "#" + filename_lvst + "_recons.obj";
				string path_pred_mesh = in_tensorflow_path + os_sep + seed_num + "#" + simname + "#" + filename_lvst + "_pred.obj";
				//string path_streamline = in_tensorflow_path + os_sep + seed_num + "#" + simname + "#" + "flipStreamline_" + frame_num + "_resampled.txt";
				string path_streamline = in_tensorflow_path + os_sep + seed_num + "#" + simname + "#" + "flipStreamline_" + frame_num + "_resampled_agglomerativeclustering_cluster_centers.txt";
				string path_mesh = in_tensorflow_path + os_sep + simname + "#" + "flip_" + frame_num + ".gz";
				//std::cout << path_test_lvst << "\n" << path_streamline << "\n" << path_streamline;
				
				// load streamline
				fstream file(path_streamline, std::fstream::in);
				ERROR_CHECK(!file.is_open(), ("ERROR: cannot open file: %s\n", path_streamline.c_str()));
				string line;
				int line_id = 0;
				while (getline(file, line, '\n'))
				{
					stringstream ssline(line);
					string token;
					int point_id = 0;
					while (getline(ssline, token, ' '))
						streamline_grid[current_frame][line_id][point_id++] = (atof(token.c_str()) - center) * scale;
					ERROR_CHECK(point_id % 3 != 0, "ERROR: streamline points not correct!\n");
					streamline_length[current_frame][line_id] = point_id / 3;
					line_id++;
					ERROR_CHECK(line_id >= max_streamline_number, "ERROR: too many streamlines!\n");
				}
				num_streamlines[current_frame] = line_id;
				file.close();

				// load mesh
				train_animation[current_frame]->Read_Manta_Mesh_OBJ(path_test_mesh.c_str());
				train_animation[current_frame]->Build_Connectivity();
				train_animation[current_frame]->Lap_Smooth_Mesh();
				train_animation[current_frame]->Build_VN(train_animation[current_frame]->X);

				predict_animation[current_frame]->Read_Manta_Mesh_OBJ(path_pred_mesh.c_str());
				predict_animation[current_frame]->Build_Connectivity();
				predict_animation[current_frame]->Lap_Smooth_Mesh();
				predict_animation[current_frame]->Build_VN(predict_animation[current_frame]->X);

				current_frame++;
			}
		}
		printf("number frames = %d\n", current_frame);

		end_frame = current_frame-1;
		Update(start_frame);
	}

	//load demo_sequence
	void Load_Tensorflow_Demo_Sequence()
	{
		string inpath = "..\\tensorflow\\demo_sequence";
		start_frame = 20;
		end_frame = 99;
		Set_Scene_Name("demo_sequence");

		// allocate memory for streamlines
		if (streamline_grid == NULL)
		{
			num_streamlines = new int[max_frames];
			streamline_length = new int*[max_frames];
			streamline_grid = new TYPE**[max_frames];
			for (int f = 0; f < max_frames; f++)
			{
				streamline_length[f] = new int[max_streamline_number];
				streamline_grid[f] = new TYPE*[max_streamline_number];
				for (int n = 0; n < max_streamline_number; n++)
					streamline_grid[f][n] = new TYPE[max_streamline_length * 3];
			}
			streamline_colors = new TYPE[3 * max_streamline_number];
		}
		float center = 0.5 * grid_res;
		float scale = 1.0 / grid_res;
		// allocate memory for meshes
		if (train_animation == NULL)
		{
			train_animation = new Tri_Mesh<TYPE>*[max_frames];
			predict_animation = new Tri_Mesh<TYPE>*[max_frames];
			for (int i = 0; i < max_frames; i++)
			{
				train_animation[i] = new Tri_Mesh<TYPE>(max_vertices);
				predict_animation[i] = new Tri_Mesh<TYPE>(max_vertices);
			}
		}

		string os_sep = "\\";
		 //../VisMat/data\\4096\\sbfs_flip_water_pipes_10113\\flipSketchGrid_0030.bin 
		 //../manta/sbfs_scenes/data\\sbfs_flip_water_pipes_10113\\flipLevelSet_0030.uni 
		 //../manta/sbfs_scenes/data\\sbfs_flip_water_pipes_10113\\flipVel_0030.uni

		printf("start loading tensorflow demo sequences\n");
		for (int current_frame = start_frame; current_frame <= end_frame; current_frame++)
		{
			char addr_lvst_c[1024];
			char addr_sketch_c[1024];

			sprintf(addr_lvst_c, string("../manta/sbfs_scenes/data\\sbfs_flip_water_pipes_10000\\flipLevelSet_%04d.uni").c_str(), current_frame);
			string addr_lvst = string(addr_lvst_c);
			string filename_lvst = addr_lvst.substr(addr_lvst.rfind(os_sep) + 1, addr_lvst.rfind('.') - addr_lvst.rfind(os_sep) - 1);
			string frame_num = filename_lvst.substr(filename_lvst.rfind('_') + 1);
			string addr_lvst_pre = addr_lvst.substr(0, addr_lvst.rfind(os_sep));
			string addr_lvst_pre_pre = addr_lvst_pre.substr(0, addr_lvst_pre.rfind(os_sep));
			string simname = addr_lvst_pre.substr(addr_lvst_pre.rfind(os_sep) + 1);

			sprintf(addr_sketch_c, string("../VisMat/data\\4096\\sbfs_flip_water_pipes_10000\\flipSketchGrid_%04d.bin").c_str(), current_frame);
			string addr_sketch = string(addr_sketch_c);
			string addr_sketch_pre = addr_sketch.substr(0, addr_sketch.rfind(os_sep));
			string addr_sketch_pre_pre = addr_sketch_pre.substr(0, addr_sketch_pre.rfind(os_sep));
			string seed_num = addr_sketch_pre_pre.substr(addr_sketch_pre_pre.rfind(os_sep) + 1);

			string path_test_mesh = inpath + os_sep + seed_num + "#" + simname + "#" + filename_lvst + "_recons.obj";
			string path_pred_mesh = inpath + os_sep + seed_num + "#" + simname + "#" + filename_lvst + "_pred.obj";
			string path_streamline = inpath + os_sep + seed_num + "#" + simname + "#" + "flipStreamline_" + frame_num + "_resampled.txt";
			string path_mesh = inpath + os_sep + simname + "#" + "flip_" + frame_num + ".gz";

			// load streamline
			fstream file(path_streamline, std::fstream::in);
			ERROR_CHECK(!file.is_open(), ("ERROR: cannot open file: %s\n", path_streamline.c_str()));
			string line;
			int line_id = 0;
			while (getline(file, line, '\n'))
			{
				stringstream ssline(line);
				string token;
				int point_id = 0;
				while (getline(ssline, token, ' '))
					streamline_grid[current_frame][line_id][point_id++] = (atof(token.c_str()) - center) * scale;
				ERROR_CHECK(point_id % 3 != 0, "ERROR: streamline points not correct!\n");
				streamline_length[current_frame][line_id] = point_id / 3;
				line_id++;
				ERROR_CHECK(line_id >= max_streamline_number, "ERROR: too many streamlines!\n");
			}
			num_streamlines[current_frame] = line_id;
			file.close();

			// load mesh
			train_animation[current_frame]->Read_Manta_Mesh_OBJ(path_test_mesh.c_str());
			train_animation[current_frame]->Build_Connectivity();
			train_animation[current_frame]->Lap_Smooth_Mesh();
			train_animation[current_frame]->Build_VN(train_animation[current_frame]->X);

			predict_animation[current_frame]->Read_Manta_Mesh_OBJ(path_pred_mesh.c_str());
			predict_animation[current_frame]->Build_Connectivity();
			predict_animation[current_frame]->Lap_Smooth_Mesh();
			predict_animation[current_frame]->Build_VN(predict_animation[current_frame]->X);
		}

		Update(start_frame);
	}

	//load demo_interpolation
	void Load_Tensorflow_Demo_Interpolate()
	{
		string inpath = "..\\tensorflow\\demo_interpolate";
		start_frame = 0;
		end_frame = 50;
		Set_Scene_Name("demo_interpolate");

		float center = 0.5 * grid_res;
		float scale = 1.0 / grid_res;
		// allocate memory for meshes
		if (train_animation == NULL)
		{
			train_animation = new Tri_Mesh<TYPE>*[max_frames];
			predict_animation = new Tri_Mesh<TYPE>*[max_frames];
			for (int i = 0; i < max_frames; i++)
			{
				train_animation[i] = new Tri_Mesh<TYPE>(max_vertices);
				predict_animation[i] = new Tri_Mesh<TYPE>(max_vertices);
			}
		}
		string os_sep = "\\";
		printf("start loading tensorflow demo interpolation\n");
		for (int current_frame = start_frame; current_frame <= end_frame; current_frame++)
		{
			float a = current_frame / 50.0;
			char path_pred_mesh[1024];
			sprintf(path_pred_mesh, string(inpath + os_sep + "test_predict_0_20_%.2f.obj").c_str(), a);
	
			// load mesh
			predict_animation[current_frame]->Read_Manta_Mesh_OBJ(path_pred_mesh);
			predict_animation[current_frame]->Build_Connectivity();
			predict_animation[current_frame]->Lap_Smooth_Mesh();
			predict_animation[current_frame]->Build_VN(predict_animation[current_frame]->X);
		}

		Update(start_frame);
	}

	// check if the levelset grid(binarified) and sketch grid exported from python get_data_next_batch()
	void Load_Sketch_Levelset_Grid_Uni(const string inPath)
	{
		start_frame = 0;
		end_frame = 19;
		Set_Scene_Name("voxelization");

		string file_sketch_acc		= "sketch_occ";
		string file_sketch_vel		= "sketch_vel";
		string file_levelset		= "levelset";
		string file_levelset_vel	= "vel";

 		ERROR_CHECK(end_frame - start_frame > max_frames, "ERROR: EXCEED MAX FRAMES");

		// allocate memory
		if (grid_sketch == NULL && grid_levelset == NULL)
		{
			grid_sketch			= new TYPE*[max_frames];
			grid_levelset		= new TYPE*[max_frames];
			grid_sketch_vel		= new TYPE*[max_frames];
			grid_levelset_vel	= new TYPE*[max_frames];
			for (int f = 0; f < max_frames; f++)
			{
				grid_sketch[f]			= new TYPE[grid_res * grid_res * grid_res];
				grid_levelset[f]		= new TYPE[grid_res * grid_res * grid_res];
				grid_sketch_vel[f]		= new TYPE[grid_res * grid_res * grid_res * 3];
				grid_levelset_vel[f]	= new TYPE[grid_res * grid_res * grid_res * 3];
			}
		}

		// load data
		PRINTF("start loading the sketch grid data, this may take a while...\n");
		for (int f = start_frame; f <= end_frame; f++)
		{
			char filename_sketch_occ[1024], filename_sketch_vel[1024], filename_levelset[1024], filename_levelset_vel[1024];
			sprintf(filename_sketch_occ, string(inPath + "\\" + file_sketch_acc + "_%04d.uni").c_str(), f);
			sprintf(filename_levelset, string(inPath + "\\" + file_levelset + "_%04d.uni").c_str(), f);
			sprintf(filename_sketch_vel, string(inPath + "\\" + file_sketch_vel + "_%04d.uni").c_str(), f);
			sprintf(filename_levelset_vel, string(inPath + "\\" + file_levelset_vel + "_%04d.uni").c_str(), f);
			Read_Grid_Uni(filename_sketch_occ, grid_sketch[f], grid_res, grid_res, grid_res, false);
			Read_Grid_Uni(filename_levelset, grid_levelset[f], grid_res, grid_res, grid_res, false);
			Read_Grid_Uni(filename_sketch_vel, grid_sketch_vel[f], grid_res, grid_res, grid_res, true);
			Read_Grid_Uni(filename_levelset_vel, grid_levelset_vel[f], grid_res, grid_res, grid_res, true);
		}
		PRINTF("end loading the grid data\n");

		Update(start_frame);
	}

	// check prediction result generated from tensorflow
	void Load_Tensorflow_Meshes(const string inPath)
	{
		ERROR_CHECK(end_frame - start_frame > max_frames, "ERROR: EXCEED MAX FRAMES");

		if (test_animation == NULL)
		{
			test_animation = new Tri_Mesh<TYPE>*[max_frames];
			predict_animation = new Tri_Mesh<TYPE>*[max_frames];
			for (int i = 0; i < max_frames; i++)
			{
				test_animation[i] = new Tri_Mesh<TYPE>(max_vertices);
				predict_animation[i] = new Tri_Mesh<TYPE>(max_vertices);
			}		
		}

		printf("start loading the mesh sequences %s, this may take a while...\n", inPath.c_str());
		for (int current_frame = start_frame; current_frame <= end_frame; current_frame++)
		{
			char in_test_filename[1024];
			char in_predict_filename[1024];

			sprintf(in_test_filename, string(inPath + "\\test_unpacked_%04d.gz").c_str(), current_frame);
			sprintf(in_predict_filename, string(inPath + "\\predict_unpacked_%04d.gz").c_str(), current_frame);

			test_animation[current_frame]->Read_Manta_Mesh_GZ(in_test_filename);
			test_animation[current_frame]->Build_Edges();
			test_animation[current_frame]->Build_VN(test_animation[current_frame]->X);

			predict_animation[current_frame]->Read_Manta_Mesh_GZ(in_predict_filename);
			predict_animation[current_frame]->Build_Edges();
			predict_animation[current_frame]->Build_VN(predict_animation[current_frame]->X);
		}
		printf("end loading the mesh sequences\n");

		Update(start_frame);
	}

	// load streamlines generated from MATLAB(normalize to unit cube around 0)
	void Load_Streamline_Grid_TXT(const string basePath, const string sceneName, const string filename, bool resampled = true)
	{
		scene_name = sceneName;
		ERROR_CHECK(end_frame - start_frame > max_frames, "ERROR: EXCEED MAX FRAMES");

		// allocate memory
		if (streamline_grid == NULL)
		{
			num_streamlines = new int[max_frames];
			streamline_length = new int*[max_frames];
			streamline_grid = new TYPE**[max_frames];
			for (int f = 0; f < max_frames; f++)
			{
				streamline_length[f] = new int[max_streamline_number];
				streamline_grid[f] = new TYPE*[max_streamline_number];
				for (int n = 0; n < max_streamline_number; n++)
					streamline_grid[f][n] = new TYPE[max_streamline_length * 3];
			}
			streamline_colors = new TYPE[3 * max_streamline_number];
		}

		float center = 0.5 * grid_res;
		float scale = 1.0 / grid_res;
		// load data
		bool streamline_label = false;
		printf("start loading the streamlines %s, this may take a while...\n", filename.c_str());
		for (int current_frame = start_frame; current_frame <= end_frame; current_frame++)
		{
			char current_filename[1024];
			if (resampled)
			{
				// clustering case
				sprintf(current_filename, string(basePath + "\\" + scene_name + "\\" + filename + "_%04d_resampled_agglomerativeclustering_cluster_centers.txt").c_str(), current_frame);
				//sprintf(current_filename, string(basePath + "\\" + scene_name + "\\" + filename + "_%04d_resampled_kmeans_cluster_centers.txt").c_str(), current_frame);
				/*sprintf(current_filename, string(basePath + "\\" + scene_name + "\\" + filename + "_%04d_resampled.txt").c_str(), current_frame);
				streamline_label = true;*/

				//no clustering
				//sprintf(current_filename, string(basePath + "\\" + scene_name + "\\" + filename + "_%04d_resampled.txt").c_str(), current_frame);
			}
			else
				sprintf(current_filename, string(basePath + "\\" + scene_name + "\\" + filename + "_%04d.txt").c_str(), current_frame);		
			//printf("%s\n", current_filename);

			fstream file(current_filename, std::fstream::in);
			ERROR_CHECK(!file.is_open(), ("ERROR: cannot open file: %s\n", current_filename));
			string line;
			int line_id = 0;
			while (getline(file, line, '\n'))
			{
				stringstream ssline(line);
				string token;
				int point_id = 0;
				while (getline(ssline, token, ' '))
				{
					streamline_grid[current_frame][line_id][point_id++] = (atof(token.c_str()) - center) * scale;
					//printf("%d, %d, %d, %f\n", current_frame, line_id, point_id, atof(token.c_str()));
				}			
				ERROR_CHECK(point_id % 3 != 0, "ERROR: streamline points not correct!\n");
				streamline_length[current_frame][line_id] = point_id / 3;
				//printf("streamline length = %d\n", streamline_length[current_frame][line_id]);
				line_id++;
				ERROR_CHECK(line_id >= max_streamline_number, "ERROR: too many streamlines!\n");
			}
			num_streamlines[current_frame] = line_id;
			file.close();
		}

		// load streamline labels if any
		if (streamline_label)
		{
			streamline_labels = new int*[max_frames];
			for (int f = 0; f < max_frames; f++)
				streamline_labels[f] = new int[max_streamline_number];

			for (int current_frame = start_frame; current_frame <= end_frame; current_frame++)
			{
				char current_filename[1024];
				//sprintf(current_filename, string(basePath + "\\" + scene_name + "\\" + filename + "_%04d_resampled_agglomerativeclustering_labels.txt").c_str(), current_frame);
				sprintf(current_filename, string(basePath + "\\" + scene_name + "\\" + filename + "_%04d_resampled_kmeans_labels.txt").c_str(), current_frame);

				FILE *fp = fopen(current_filename, "r+");
				if (fp == 0) { printf("ERROR: cannot open %s\n", current_filename); getchar(); }
				for (int l = 0; l < num_streamlines[current_frame]; l++)
				{
					fscanf(fp, "%d", &streamline_labels[current_frame][l]);
				}
				fclose(fp);
			}
		}


		Update(start_frame);
	}

	// load voxlized streamlines generated from MATLAB
	void Load_Sketch_Grid_Bin(const string basePath, const string sceneName, const string filename)
	{
		scene_name = sceneName;
		ERROR_CHECK(end_frame - start_frame > max_frames, "ERROR: EXCEED MAX FRAMES");

		// allocate memory
		if (grid_sketch == NULL)
		{
			grid_sketch = new TYPE*[max_frames];
			for (int f = 0; f < max_frames; f++)
				grid_sketch[f] = new TYPE[grid_res * grid_res * grid_res];
		}

		// load data
		PRINTF("start loading the sketch grid data %s, this may take a while...\n", filename.c_str());
 		for (int f = start_frame;	f <= end_frame; f++)
		{
			char current_filename[1024];
			//sprintf(current_filename, string(basePath + "\\" + scene_name + "\\" + filename + "_%04d.bin").c_str(), f);
			sprintf(current_filename, string(basePath + "\\" + scene_name + "\\" + filename + "_%04d_agglomerativeclustering_cluster_centers.bin").c_str(), f);
			Read_Grid_Bin(current_filename, grid_sketch[f], grid_res, grid_res, grid_res);
		}
		PRINTF("end loading the grid data\n");

		Update(start_frame);
	}

	// load particles generated from mantaflow
	void Load_Flip_Parts_TXT(const string basePath, const string sceneName, const string filename)
	{
		ERROR_CHECK(end_frame - start_frame > max_frames, "ERROR: EXCEED MAX FRAMES");

		scene_name = sceneName;
		if (pathline_animation == NULL)
		{
			pathline_animation = new Tri_Mesh<TYPE>*[max_frames];;
			for (int i = 0; i < max_frames; i++)
				pathline_animation[i] = new Tri_Mesh<TYPE>(max_particles);
		}

		printf("start loading the particle sequences %s, this may take a while...\n", filename.c_str());
		for (int current_frame = start_frame; current_frame <= end_frame; current_frame++)
		{
			char current_filename[1024];
			sprintf(current_filename, string(basePath + "\\" + scene_name + "\\" + filename + "_%04d.txt").c_str(), current_frame);
			pathline_animation[current_frame]->Read_Manta_Parts_TXT(current_filename);
		}
		printf("end loading the particle sequences\n");

		Update(start_frame);
	}
	
	// load levelset meshes recontructed from mantaflow
	void Load_Manta_Meshes(const string inbasePath, const string outbasePath, const string sceneName, const string filename, 
		bool compressed = true, bool exportOrientedPoints = false)
	{
		ERROR_CHECK(end_frame - start_frame > max_frames, "ERROR: EXCEED MAX FRAMES");

		std::string out_dir = outbasePath + "\\" + sceneName;
		Create_Directory(out_dir);

		scene_name = sceneName;
		if (lvst_mesh_animation == NULL)
		{
			lvst_mesh_animation = new Tri_Mesh<TYPE>*[max_frames];;
			for (int i = 0; i < max_frames; i++)
				lvst_mesh_animation[i] = new Tri_Mesh<TYPE>(max_vertices);
		}

		printf("start loading the mesh sequences %s, this may take a while...\n", filename.c_str());
		for (int current_frame = start_frame; current_frame <= end_frame; current_frame++)
		{
			char in_filename[1024];
			char out_filename_ply[1024];
			char out_filename_xyz[1024];
			if (compressed)
			{
				sprintf(in_filename, string(inbasePath + "\\" + scene_name + "\\" + filename + "_%04d.gz").c_str(), current_frame);
				lvst_mesh_animation[current_frame]->Read_Manta_Mesh_GZ(in_filename);
			}
			else
			{
				sprintf(in_filename, string(inbasePath + "\\" + scene_name + "\\" + filename + "_%04d.obj").c_str(), current_frame);
				lvst_mesh_animation[current_frame]->Read_Manta_Mesh_OBJ(in_filename);
			}
			lvst_mesh_animation[current_frame]->Build_Edges();
			lvst_mesh_animation[current_frame]->Build_VN(lvst_mesh_animation[current_frame]->X);
			if (exportOrientedPoints)
			{
				sprintf(out_filename_xyz, string(out_dir + "\\" + filename + "_%04d.point.xyz").c_str(), current_frame);
				//lvst_mesh_animation[current_frame]->Write_Oriented_Points_xyz(out_filename);
				lvst_mesh_animation[current_frame]->Write_Points_xyz(out_filename_xyz);

				sprintf(out_filename_ply, string(out_dir + "\\" + filename + "_%04d.orientedPoints.ply").c_str(), current_frame);
#ifdef USE_CGAL
				//Write_PLY_Point_Set(out_filename_ply, \
					lvst_mesh_animation[current_frame]->X, lvst_mesh_animation[current_frame]->VN, lvst_mesh_animation[current_frame]->number);
				Write_PLY_Point_Set(out_filename_xyz, out_filename_ply);
#endif
			}
		}
		printf("end loading the mesh sequences\n");

		Update(start_frame);
	}

	// surface reconstruction from point set
#ifdef USE_CGAL
	
	void Oriented_Points_Surface_Reconstruction(const string inbasePath, const string outbasePath, const string sceneName, const string filename)
	{
		ERROR_CHECK(end_frame - start_frame > max_frames, "ERROR: EXCEED MAX FRAMES");

		std::string out_dir = outbasePath + "\\" + sceneName;
		Create_Directory(out_dir);

		scene_name = sceneName;
		printf("start reconstruction the mesh sequences %s, this may take a while...\n", filename.c_str());
		for (int current_frame = start_frame; current_frame <= end_frame; current_frame++)
		{
			char filename_base[1024];
			sprintf(filename_base, string(filename + "_%04d").c_str(), current_frame);
			string in_dir = string(inbasePath + "\\" + scene_name);
			string out_dir = string(outbasePath + "\\" + scene_name);
			string filename_base_str = string(filename_base);

			printf("\t reconstructing %s...\n", filename_base_str.c_str());
			//CGAL_Poisson_Reconstruction_Function(in_dir, out_dir, filename_base_str);
			//CGAL_Poisson_Reconstruction_Class(in_dir, out_dir, filename_base_str);
			Adaptive_Multigrid_Solvers_Surface_Reconstruction(in_dir, out_dir, filename_base_str);
		}

		printf("end reconstruction the mesh sequences\n");

		Update(start_frame);
	}

	
	void Load_Reconstruction_Meshes(const string inbasePath, const string sceneName, const string filename)
	{
		ERROR_CHECK(end_frame - start_frame > max_frames, "ERROR: EXCEED MAX FRAMES");

		scene_name = sceneName;
		if (lvst_mesh_animation == NULL)
		{
			lvst_mesh_animation = new Tri_Mesh<TYPE>*[max_frames];;
			for (int i = 0; i < max_frames; i++)
				lvst_mesh_animation[i] = new Tri_Mesh<TYPE>(max_vertices);
		}

		printf("start loading the mesh sequences %s, this may take a while...\n", filename.c_str());
		for (int current_frame = start_frame; current_frame <= end_frame; current_frame++)
		{
			char in_filename[1024];
			//sprintf(in_filename, string(inbasePath + "\\" + scene_name + "\\" + filename + "_%04d.PoissonRecon.ply").c_str(), current_frame);
			sprintf(in_filename, string(inbasePath + "\\" + scene_name + "\\" + filename + "_%04d.SSDRecon.ply").c_str(), current_frame);
			Read_PLY_File(string(in_filename),
				lvst_mesh_animation[current_frame]->X, lvst_mesh_animation[current_frame]->number,
				lvst_mesh_animation[current_frame]->T, lvst_mesh_animation[current_frame]->t_number);

			lvst_mesh_animation[current_frame]->Build_Edges();
			lvst_mesh_animation[current_frame]->Build_VN(lvst_mesh_animation[current_frame]->X, false);
		}
		printf("end loading the mesh sequences\n");

		Update(start_frame);
	}

#endif

	// build pathline from imported particles(normalize to unit cube around 0)
	void Build_PathLines()
	{
		num_pathlines = pathline_animation[start_frame]->number;
		float center = 0.5 * grid_res;
		float scale = 1.0/ grid_res;

		pathline_particle = new TYPE*[num_pathlines];
		pathcolors = new TYPE[3 * num_pathlines];
		for (int i = 0; i < num_pathlines; i++)
		{
			pathline_particle[i] = new TYPE[3 * max_frames];
			for (int t = start_frame; t <= end_frame; t++)
			{
				pathline_particle[i][3 * t + 0] = (pathline_animation[t]->X[3 * i + 0] - center) * scale;
				pathline_particle[i][3 * t + 1] = (pathline_animation[t]->X[3 * i + 1] - center) * scale;
				pathline_particle[i][3 * t + 2] = (pathline_animation[t]->X[3 * i + 2] - center) * scale;
			}

			// assign random color
			pathcolors[3 * i + 0] = ((float)rand() / (RAND_MAX));
			pathcolors[3 * i + 1] = ((float)rand() / (RAND_MAX));
			pathcolors[3 * i + 2] = ((float)rand() / (RAND_MAX));
		}
	}

	void Export_Pathlines(const string basePath, const string sceneName)
	{
		string filename = basePath + "\\" + sceneName + ".txt";
		FILE *fp = fopen(filename.c_str(), "w+");
		if (fp == 0) { printf("ERROR: cannot open %s\n", filename); getchar(); }

		// each line contains the trajectory for each particle
		int num_particles = pathline_animation[start_frame]->number;
		for (int i = 0; i < num_particles; i++)
		{
			int t;
			for (t = start_frame; t < end_frame; t++)
				fprintf(fp, "%f,%f,%f,", pathline_particle[i][3 * t + 0], pathline_particle[i][3 * t + 1], pathline_particle[i][3 * t + 2]);
			fprintf(fp, "%f,%f,%f\n", pathline_particle[i][3 * t + 0], pathline_particle[i][3 * t + 1], pathline_particle[i][3 * t + 2]);
		}

		fclose(fp);
	}

	// note the imported pathlines do not necessarily start from the first frame, it depends on the export function!
	void Import_Pathlines(const string basePath, const string sceneName, bool clustered = false, int num_clusters = 4)
	{
		scene_name = sceneName;
		if (clustered)
			if (num_clusters < 10)
				scene_name = scene_name + "_cluster_00" + to_string(num_clusters);
			else if (num_clusters < 100)
				scene_name = scene_name + "_cluster_0" + to_string(num_clusters);
			else
				scene_name = scene_name + "_cluster_" + to_string(num_clusters);
		string filename = basePath + "\\" + scene_name + ".txt";

		vector<vector<float>> dataMatrix;
		fstream file;
		string line;
		file.open(filename);
		if (!file.is_open()) { printf("ERROR: cannot open %s\n", filename.c_str()); getchar(); }
		while (getline(file, line, '\n'))
		{
			vector<float> dataLine;
			stringstream ssline(line);
			string token;
			while (getline(ssline, token, ','))
				dataLine.push_back(atof(token.c_str()));
			if (dataLine.size() % 3 != 0)
			{
				printf("Error! data not contains xyz\n"); getchar();
			}
			dataMatrix.push_back(dataLine);
		}
		file.close();

		num_pathlines = dataMatrix.size();
		ERROR_CHECK((end_frame - start_frame + 1 != dataMatrix[0].size() / 3), "Error in frame range!\n");

		pathline_particle = new TYPE*[num_pathlines];
		pathcolors = new TYPE[3 * num_pathlines];
		for (int i = 0; i < num_pathlines; i++)
		{
			pathline_particle[i] = new TYPE[3 * max_frames];
			for (int t = start_frame; t <= end_frame; t++) // offset the start_frame in dataMatrix, which always starts from zero
			{
				pathline_particle[i][3 * t + 0] = dataMatrix[i][3 * (t - start_frame) + 0];
				pathline_particle[i][3 * t + 1] = dataMatrix[i][3 * (t - start_frame) + 1];
				pathline_particle[i][3 * t + 2] = dataMatrix[i][3 * (t - start_frame) + 2];
			}

			// assign random color
			pathcolors[3 * i + 0] = ((float)rand() / (RAND_MAX));
			pathcolors[3 * i + 1] = ((float)rand() / (RAND_MAX));
			pathcolors[3 * i + 2] = ((float)rand() / (RAND_MAX));
		}

		printf("number paths = %d, number frames = %d\n", num_pathlines, end_frame);

		Update(start_frame);
	}
	

	// --- rendering functions
	void Draw_Pathlines(TYPE linewidth = 1.0)
	{
		if (pathline_particle == NULL) return;

		glUseProgram(0);
		glDisable(GL_LIGHTING);

		glLineWidth(linewidth);
		glEnable(GL_LINE_SMOOTH);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glDisable(GL_BLEND);
		for (int p = 0; p < num_pathlines; p++) // particles
		{
			glColor4f(pathcolors[3 * p + 0], pathcolors[3 * p + 1], pathcolors[3 * p + 2], 0.4);
			glBegin(GL_LINE_STRIP);
			for (int t = draw_start_frame; t <= draw_end_frame; t++) // paths
				glVertex3f(pathline_particle[p][3 * t + 0], pathline_particle[p][3 * t + 1], pathline_particle[p][3 * t + 2]);
			glEnd();
		}

		glDisable(GL_BLEND);
		glEnable(GL_LIGHTING);
	}

	int Draw_Streamlines(TYPE linewidth = 1.0)
	{
		if (streamline_grid == NULL) return 0;
		if (cur_frame < start_frame || cur_frame > end_frame) return 0;
		//printf("current frame = %d, start_frame = %d, end_frame = %d\n", cur_frame, start_frame, end_frame);

		static bool set_color = false;
		static int last_frame = -1;
		static std::vector<float> colors;
		if (!set_color)
		{
			for (int i = 0; i < max_streamline_number; i++)
			{
				colors.push_back((float)rand() / (RAND_MAX));
				colors.push_back((float)rand() / (RAND_MAX));
				colors.push_back((float)rand() / (RAND_MAX));
			}
			set_color = true;
		}

		// count number of distinct labels
		if (streamline_labels)
		{
			std::unordered_set<int> set;
			int num_labels = 0;
			for (int i = 0; i < num_streamlines[cur_frame]; i++)
				if (set.find(streamline_labels[cur_frame][i]) == set.end())
				{
					set.insert(streamline_labels[cur_frame][i]);
					num_labels++;
				}
			for (int i = 0; i < num_streamlines[cur_frame]; i++)
			{
				streamline_colors[3 * i + 0] = colors[3 * streamline_labels[cur_frame][i] + 0];
				streamline_colors[3 * i + 1] = colors[3 * streamline_labels[cur_frame][i] + 1];
				streamline_colors[3 * i + 2] = colors[3 * streamline_labels[cur_frame][i] + 2];
			}
		}
		else
		{
			for (int i = 0; i < num_streamlines[cur_frame]; i++)
			{
				streamline_colors[3 * i + 0] = colors[3 * i + 0];
				streamline_colors[3 * i + 1] = colors[3 * i + 1];
				streamline_colors[3 * i + 2] = colors[3 * i + 2];
			}
		}

		glUseProgram(0);
		glDisable(GL_LIGHTING);
		glLineWidth(linewidth);
		glEnable(GL_LINE_SMOOTH);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glDisable(GL_BLEND);
		for (int line = 0; line < num_streamlines[cur_frame]; line++) // streamline at current frame
		{
			glColor4f(streamline_colors[3 * line + 0], streamline_colors[3 * line + 1], streamline_colors[3 * line + 2], 1.0);
			//glColor4f(0.2,0.2,0.2, 0.6);
			glBegin(GL_LINE_STRIP);
			for (int p = 0; p < streamline_length[cur_frame][line]; p++)
				glVertex3f(streamline_grid[cur_frame][line][3 * p + 0], streamline_grid[cur_frame][line][3 * p + 1], streamline_grid[cur_frame][line][3 * p + 2]);
			glEnd();
		}
		glDisable(GL_BLEND);
		glEnable(GL_LIGHTING);

		return num_streamlines[cur_frame];
	}

	void Arrow(GLdouble x1, GLdouble y1, GLdouble z1, GLdouble x2, GLdouble y2, GLdouble z2, GLdouble D)
	{
		double x = x2 - x1;
		double y = y2 - y1;
		double z = z2 - z1;
		double L = sqrt(x*x + y*y + z*z);

		GLUquadricObj *quadObj;

		glPushMatrix();

		glTranslated(x1, y1, z1);

		if ((x != 0.) || (y != 0.)) {
			glRotated(atan2(y, x) / 0.0174533, 0., 0., 1.);
			glRotated(atan2(sqrt(x*x + y*y), z) / 0.0174533, 0., 1., 0.);
		}
		else if (z<0) {
			glRotated(180, 1., 0., 0.);
		}

		glTranslatef(0, 0, L - 4 * D);

		quadObj = gluNewQuadric();
		gluQuadricDrawStyle(quadObj, GLU_FILL);
		gluQuadricNormals(quadObj, GLU_SMOOTH);
		gluQuadricOrientation(quadObj, GLU_OUTSIDE);
		gluCylinder(quadObj, 2 * D, 0.0, 4 * D, 32, 1);
		gluDeleteQuadric(quadObj);

		quadObj = gluNewQuadric();
		gluQuadricDrawStyle(quadObj, GLU_FILL);
		gluQuadricNormals(quadObj, GLU_SMOOTH);
		gluQuadricOrientation(quadObj, GLU_OUTSIDE);
		gluDisk(quadObj, 0.0, 2 * D, 32, 1);
		gluDeleteQuadric(quadObj);

		glTranslatef(0, 0, -L + 4 * D);

		quadObj = gluNewQuadric();
		gluQuadricDrawStyle(quadObj, GLU_FILL);
		gluQuadricNormals(quadObj, GLU_SMOOTH);
		gluQuadricOrientation(quadObj, GLU_OUTSIDE);
		gluCylinder(quadObj, D, D, L - 4 * D, 32, 1);
		gluDeleteQuadric(quadObj);

		quadObj = gluNewQuadric();
		gluQuadricDrawStyle(quadObj, GLU_FILL);
		gluQuadricNormals(quadObj, GLU_SMOOTH);
		gluQuadricOrientation(quadObj, GLU_OUTSIDE);
		gluDisk(quadObj, 0.0, D, 32, 1);
		gluDeleteQuadric(quadObj);

		glPopMatrix();

	}

	void Draw_Arrow(int mode)
	{
		if (grid_sketch == NULL || grid_levelset == NULL) return;
		if (cur_frame < start_frame || cur_frame > end_frame) return;

		float center = 0.5 * grid_res;
		float scale = 1.0 / grid_res;

		GLfloat color[] = { 1.f, .0f, 0.f, 1.f };
		glUseProgram(0);
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glEnable(GL_DEPTH_TEST);
		glShadeModel(GL_SMOOTH);

		TYPE vel[3], vel_mag;
		FOR_EVERY_CELL
		{
			TYPE p[3] = { i, j, k };
			p[0] = (p[0] - center) * scale;
			p[1] = (p[1] - center) * scale;
			p[2] = (p[2] - center) * scale;
			glPushMatrix();
			glTranslated(0, 0, 0); //no need translation here, arrow function would do it
			if (mode == 0)
			{
				vel[0] = grid_sketch_vel[cur_frame][3 * id + 0];
				vel[1] = grid_sketch_vel[cur_frame][3 * id + 1];
				vel[2] = grid_sketch_vel[cur_frame][3 * id + 2];
				vel_mag = sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
			}
			else if (mode == 1)
			{
				vel[0] = grid_levelset_vel[cur_frame][3 * id + 0];
				vel[1] = grid_levelset_vel[cur_frame][3 * id + 1];
				vel[2] = grid_levelset_vel[cur_frame][3 * id + 2];
				vel_mag = sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
			}
			if (vel_mag > 1e-8) {
				vel[0] /= vel_mag;
				vel[1] /= vel_mag;
				vel[2] /= vel_mag;

				glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
				Arrow(p[0], p[1], p[2], p[0] + vel[0] * scale * 2.5, p[1] + vel[1] * scale * 2.5, p[2] + vel[2] * scale * 2.5, 0.1 * scale * 2.5);
			}
			glPopMatrix();
		}

	}

	void Draw_Voxels(int mode = 0)
	{
		if (grid_sketch == NULL || grid_levelset == NULL) return;
		if (cur_frame < start_frame || cur_frame > end_frame) return;

		float center = 0.5 * grid_res;
		float scale = 1.0 / grid_res;

		GLfloat color[] = { 0.f, .8f, .8f, 0.5f };
		glUseProgram(0);
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glEnable(GL_DEPTH_TEST);
		glShadeModel(GL_SMOOTH);
		//glEnable(GL_BLEND);
		//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glLineWidth(2.0);
		FOR_EVERY_CELL
		{
			TYPE p[3] = { i, j, k };
			p[0] = (p[0] - center) * scale;
			p[1] = (p[1] - center) * scale;
			p[2] = (p[2] - center) * scale;
			glPushMatrix();
			glTranslated(p[0], p[1], p[2]);
			if (mode == 0 && grid_sketch[cur_frame][id])
			{
				glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
				//The cube is centered at the modeling coordinates origin with sides of length size
				glutWireCube(scale*1.01);
				glutSolidCube(scale); 		

				/*GLUquadricObj *quadObj = gluNewQuadric();
				gluQuadricDrawStyle(quadObj, GLU_FILL);
				gluQuadricNormals(quadObj, GLU_SMOOTH);
				gluQuadricOrientation(quadObj, GLU_OUTSIDE);
				gluSphere(quadObj, scale*0.5, 32, 32);
				gluDeleteQuadric(quadObj);*/
			}
			else if (mode == 1 && grid_levelset[cur_frame][id] > 0.5) // check binary levelset data 
			{
				glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
				glutWireCube(scale*1.01);
				glutSolidCube(scale);
			}
			glPopMatrix();
		}
		glDisable(GL_BLEND);
	}

	void Draw_Sketches()
	{
		if (grid_sketch == NULL) return;
		if (cur_frame < start_frame || cur_frame > end_frame) return;

		float center = 0.5 * grid_res;
		float scale = 1.0 / grid_res;

		glUseProgram(0);
		glDisable(GL_LIGHTING);

		FOR_EVERY_CELL
		{
			TYPE p[3] = {i, j, k};
			p[0] = (p[0] - center) * scale;
			p[1] = (p[1] - center) * scale;
			p[2] = (p[2] - center) * scale;
			glPushMatrix();
			glTranslated(p[0], p[1], p[2]);
			if (grid_sketch[cur_frame][id])
			{
				//float r = CLAMP(grid_sketch[cur_frame][id] / 1.0, 0.0, 1.0);
				glColor3f(1.0, 0.0, 0.0);
				glutSolidCube(scale); //The cube is centered at the modeling coordinates origin with sides of length size
				//glutWireCube(scale);
			}
			glPopMatrix();
		}

		glEnable(GL_LIGHTING);
	}

	void Draw_Edges()
	{
		if (lvst_mesh_animation == NULL) return;
		if (cur_frame < start_frame || cur_frame > end_frame) return;

		Tri_Mesh<TYPE> *mesh = lvst_mesh_animation[cur_frame];
		TYPE *X = mesh->X;
		TYPE *TN = mesh->TN;
		TYPE *VN = mesh->VN;
		int	*E = mesh->E;
		int *T = mesh->T;

		glUseProgram(0);
		glDisable(GL_LIGHTING);
		glLineWidth(0.5);
		glEnable(GL_LINE_SMOOTH);
		glEnable(GL_BLEND); // draw the mesh last to have the blending effect
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		for (int e = 0; e < mesh->e_number; e++)
		{
			glColor4f(0.2, 0.2, 0.2, 0.6);
			glBegin(GL_LINES);
			glVertex3d(X[E[e * 2 + 0] * 3 + 0], X[E[e * 2 + 0] * 3 + 1], X[E[e * 2 + 0] * 3 + 2]);
			glVertex3d(X[E[e * 2 + 1] * 3 + 0], X[E[e * 2 + 1] * 3 + 1], X[E[e * 2 + 1] * 3 + 2]);
			glEnd();
		}

		glDisable(GL_BLEND);
		glEnable(GL_LIGHTING);
	}

	void Draw_Edges(Tri_Mesh<TYPE> **&mesh_animation)
	{
		if (mesh_animation == NULL) return;
		if (cur_frame < start_frame || cur_frame > end_frame) return;

		Tri_Mesh<TYPE> *mesh = mesh_animation[cur_frame];
		TYPE *X = mesh->X;
		TYPE *TN = mesh->TN;
		TYPE *VN = mesh->VN;
		int	*E = mesh->E;
		int *T = mesh->T;

		glUseProgram(0);
		glDisable(GL_LIGHTING);
		glLineWidth(0.5);
		glEnable(GL_LINE_SMOOTH);
		glEnable(GL_BLEND); // draw the mesh last to have the blending effect
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		for (int e = 0; e < mesh->e_number; e++)
		{
			glColor4f(0.2, 0.2, 0.2, 0.6);
			glBegin(GL_LINES);
			glVertex3d(X[E[e * 2 + 0] * 3 + 0], X[E[e * 2 + 0] * 3 + 1], X[E[e * 2 + 0] * 3 + 2]);
			glVertex3d(X[E[e * 2 + 1] * 3 + 0], X[E[e * 2 + 1] * 3 + 1], X[E[e * 2 + 1] * 3 + 2]);
			glEnd();
		}

		glDisable(GL_BLEND);
		glEnable(GL_LIGHTING);
	}

	void Draw_Vertices(float pointSize = 0.001)
	{
		if (lvst_mesh_animation == NULL) return;
		if (cur_frame < start_frame || cur_frame > end_frame) return;

		Tri_Mesh<TYPE> *mesh = lvst_mesh_animation[cur_frame];
		TYPE *X = mesh->X;
		TYPE *TN = mesh->TN;
		TYPE *VN = mesh->VN;
		int	*E = mesh->E;
		int *T = mesh->T;

		glUseProgram(0);
		glDisable(GL_LIGHTING);
		glColor3f(1, 0, 0);
		for (int v = 0; v<mesh->number; v++)
		{
			glPushMatrix();
			glTranslatef(X[v * 3 + 0], X[v * 3 + 1], X[v * 3 + 2]);
			glutSolidSphere(pointSize, 10, 10);
			glPopMatrix();
		}
		glEnable(GL_LIGHTING);
	}

	void Set_Current_Frame(Tri_Mesh<TYPE> **&mesh_animation)
	{
		if (mesh_animation == NULL) return;
		if (cur_frame < start_frame || cur_frame > end_frame) return;
		Tri_Mesh<TYPE> *mesh = mesh_animation[cur_frame];


		Tri_Mesh<TYPE>::number = mesh->number;
		Tri_Mesh<TYPE>::t_number = mesh->t_number;
		Tri_Mesh<TYPE>::e_number = mesh->e_number;

		Tri_Mesh<TYPE>::X = mesh->X;
		Tri_Mesh<TYPE>::E = mesh->E;
		Tri_Mesh<TYPE>::T = mesh->T;
		Tri_Mesh<TYPE>::TN = mesh->TN;
		Tri_Mesh<TYPE>::VN = mesh->VN;
		Tri_Mesh<TYPE>::VC = mesh->VC;
	}

	// --- utiliy functions
	const string Get_Scene_Name()
	{
		return scene_name;
	}

	void Set_Scene_Name(const string &sceneName)
	{
		scene_name = sceneName;
	}

	void Create_Directory(const std::string &dir)
	{
		struct stat st = { 0 };
		if (stat(dir.c_str(), &st) == -1)
			CreateDirectory(dir.c_str(), NULL);
	}

//  --- I/O functions
	
	template<class T> // convert grid file(levelset/vel) from compressed to binary format
	void Batch_Convert_Grid_Bin(T* &grid, int sizeX, int sizeY, int sizeZ, bool XYZ,
		const string &inbasePath, const string &outbasePath, const string &sceneName, const string &filename, const string& suffix)
	{
		ERROR_CHECK(end_frame - start_frame > max_frames, "exceed max frames, please increase the number\n ");

		std::string out_dir = outbasePath + "\\" + sceneName;
		Create_Directory(out_dir);

		scene_name = sceneName;

		PRINTF("start loading the grid data %s, this may take a while...\n", filename.c_str());
		for (int current_frame = start_frame; current_frame <= end_frame; current_frame++)
		{
			char in_filename[1024];
			char out_filename[1024];
			sprintf(in_filename, string(inbasePath + "\\" + scene_name + "\\" + filename + "_%04d" + suffix).c_str(), current_frame);
			sprintf(out_filename, string(out_dir + "\\" + filename + "_%04d.bin").c_str(), current_frame);

			if (suffix == ".raw")
				Read_Grid_Raw(in_filename, grid, sizeX, sizeY, sizeZ, XYZ);
			else if (suffix == ".uni")
				Read_Grid_Uni(in_filename, grid, sizeX, sizeY, sizeZ, XYZ);
			else
				ERROR_CHECK(true, "Unknown file type\n");

			Write_Grid_Bin(out_filename, grid, sizeX, sizeY, sizeZ, XYZ);
		}
		PRINTF("end loading the grid data\n");
	}

	template<class T>
	void Read_Grid_Raw(const string& filename, T* &grid, int sizeX, int sizeY, int sizeZ, bool XYZ/*3d data*/)
	{
		gzFile gzf = gzopen(filename.c_str(), "rb");
		ERROR_CHECK(!gzf, ("can't open file %s \n", filename.c_str()));

		int bytesPerElement = sizeof(T);
		int elementDimensions = 1;
		if (XYZ)
		{
			bytesPerElement *= 3;
			elementDimensions = 3;
		}

		if (grid == NULL) grid = new T[sizeX * sizeY * sizeZ * elementDimensions];
		int bytes = bytesPerElement*sizeX*sizeY*sizeY;
		int readBytes = gzread(gzf, grid, bytes);
		ERROR_CHECK(bytes != readBytes, "can't read raw file, stream length does not match\n")
			gzclose(gzf);
	}

	template<class T>
	void Read_Grid_Uni(const string& filename, T* &grid, int sizeX, int sizeY, int sizeZ, bool XYZ/*3d data*/)
	{
		gzFile gzf = gzopen(filename.c_str(), "rb");
		ERROR_CHECK(!gzf, ("can't open file %s \n", filename.c_str()));

		char ID[5] = { 0,0,0,0,0 };
		gzread(gzf, ID, 4);

		if (!strcmp(ID, "MNT3")) {
			// current file format
			UniHeader head;
			ERROR_CHECK(gzread(gzf, &head, sizeof(UniHeader)) != sizeof(UniHeader), "can't read file, no header present");
			ERROR_CHECK(head.dimX != sizeX && head.dimY != sizeY && head.dimZ != sizeZ, "grid dim doesn't match\n");

			int bytesPerElement = sizeof(T);
			int elementDimensions = 1;
			if (XYZ)
			{
				bytesPerElement *= 3;
				elementDimensions = 3;
			}
			ERROR_CHECK(head.bytesPerElement != bytesPerElement, "grid element size doesn't match \n");

			if (grid == NULL) grid = new T[sizeX * sizeY * sizeZ * elementDimensions];
			gzread(gzf, grid, bytesPerElement*head.dimX*head.dimY*head.dimZ);
		}
		else {
			ERROR_CHECK(true, "Unknown header\n");
		}
		gzclose(gzf);
	}

	template<class T>
	void Read_Grid_Bin(const string& filename, T* &grid, int sizeX, int sizeY, int sizeZ, bool XYZ = false /*3d data*/)
	{
		ifstream file(filename.c_str(), ios::in | ios::binary);
		ERROR_CHECK(!file.is_open(), ("ERROR: cannot open %s\n", filename.c_str()));
 
		if (grid == NULL) grid = new T[sizeX * sizeY * sizeZ];
		int bytesPerElement = sizeof(T);
		if (XYZ) bytesPerElement *= 3;
		int bytes = bytesPerElement*sizeX*sizeY*sizeY;
		file.read((char*)grid, bytes);
		file.close();
	}

	template<class T>
	void Write_Grid_Bin(const string& filename, T* &grid, int sizeX, int sizeY, int sizeZ, bool XYZ/*3d data*/)
	{
		ofstream file(filename.c_str(), ios::out | ios::binary);
		ERROR_CHECK(!file.is_open(), ("ERROR: cannot open %s\n", filename.c_str()));
		ERROR_CHECK(grid == NULL, ("ERROR: grid is NULL\n"));

		int bytesPerElement = sizeof(T);
		if (XYZ) bytesPerElement *= 3;
		int bytes = bytesPerElement*sizeX*sizeY*sizeY;
		file.write((const char*)grid, bytes);
		file.close();
	}

	static const int PartSysSize = sizeof(float) * 3 + sizeof(int);
	template<class T>
	void Read_Part_Uni(const string& filename, T* &X)
	{
		gzFile gzf = gzopen(filename.c_str(), "rb");
		ERROR_CHECK(!gzf, ("can't open file %s \n", filename.c_str()));

		char ID[5] = { 0,0,0,0,0 };
		gzread(gzf, ID, 4);

		if (!strcmp(ID, "PB02"))
		{ // current file format
			UniPartHeader head;
			ERROR_CHECK(gzread(gzf, &head, sizeof(UniPartHeader)) != sizeof(UniPartHeader), "can't read file, no header present");
			ERROR_CHECK(head.bytesPerElement != PartSysSize && head.elementType != 0, "particle type doesn't match\n");

			if(X == NULL) X = new T[sizeof(T) * 3];

		}
		else {
			ERROR_CHECK(true, "Unknown header\n");
		}
		gzclose(gzf);
	}
};

#endif // !__EXAMPLE__
