#ifndef __OPENGL_MESH_h__
#define __OPENGL_MESH_h__

#define MY_INFINITE 999999999
#include <math.h>

template<class TYPE>
class Tri_Mesh
{

public:
	int		max_number;
	int		number;
	int		t_number;

	TYPE*	X;
	TYPE*	M;
	int*	T;
	TYPE*	VN;
	TYPE*	VC;
	TYPE*	TN;

	// Edge Connectivity
	int		e_number;
	int*	E;
	int*	ET;	//Triangle lists of each edge (-1 means empty)
	int*	TE;	//Edge lists of each triangle
	// Vertex connectivity
	int*	vt_num;
	// VV/VE neighborhood
	int*	VV;
	int*	VE;
	int*	vv_num;
	// TT neighborhood
	int*	TT;

	// Boundary information
	int		*boundary;

public:
	Tri_Mesh(int _max_number = 100000)
	{
		max_number	= _max_number;
		number		= 0;
		t_number	= 0;
		e_number	= 0;

		X = new TYPE[max_number * 3];
		M = new TYPE[max_number];
		T = new int[max_number * 2 * 3];
		VN = new TYPE[max_number * 3];
		VC = new TYPE[max_number * 3];
		TN = new TYPE[max_number * 2 * 3];

		E		= NULL;
		ET		= NULL;
		TE		= NULL;
		vt_num	= NULL;

		vv_num	= NULL;
		VV		= NULL;
		VE		= NULL;
		TT		= NULL;

		boundary = NULL;

		for (int i = 0; i<max_number; i++)	M[i] = 1;
		for (int i = 0; i < max_number; i++)
		{
			VC[3 * i + 0] = 0.6;
			VC[3 * i + 1] = 0.6;
			VC[3 * i + 2] = 0.6;
		}
	}

	~Tri_Mesh()
	{
		if (X)		delete[] X;
		if (T)		delete[] T;
		if (TN)		delete[] TN;
		if (VN)		delete[] VN;
		if (VC)		delete[] VC;

		if (E)		delete[] E;
		if (ET)		delete[] ET;
		if (TE)		delete[] TE;
		if (vt_num)	delete[] vt_num;

		if (VV)		delete[] VV;
		if (VE)		delete[] VE;
		if (vv_num)	delete[] vv_num;

		if (TT)		delete[] TT;
	}

	void Initialize()
	{
		//Read_OBJ("Model/ball.obj");
		//Build_Connectivity();
	}

	void Build_VN(TYPE *X, bool invertNormal = true)
	{
		memset(TN, 0, sizeof(TYPE)*number * 3);
		for (int i = 0; i<t_number; i++)
		{
			TYPE *p0 = &X[T[i * 3 + 0] * 3];
			TYPE *p1 = &X[T[i * 3 + 1] * 3];
			TYPE *p2 = &X[T[i * 3 + 2] * 3];
			if (p0 == p1 || p0 == p2 || p1 == p2)	continue;
			TYPE p10[3], p20[3];
			p10[0] = p1[0] - p0[0];
			p10[1] = p1[1] - p0[1];
			p10[2] = p1[2] - p0[2];
			p20[0] = p2[0] - p0[0];
			p20[1] = p2[1] - p0[1];
			p20[2] = p2[2] - p0[2];
			Cross(p10, p20, &TN[i * 3]);
		}

		memset(VN, 0, sizeof(TYPE)*number * 3);
		for (int i = 0; i<t_number; i++)
		{
			int v0 = T[i * 3 + 0];
			int v1 = T[i * 3 + 1];
			int v2 = T[i * 3 + 2];
			VN[v0 * 3 + 0] += TN[i * 3 + 0];
			VN[v0 * 3 + 1] += TN[i * 3 + 1];
			VN[v0 * 3 + 2] += TN[i * 3 + 2];
			VN[v1 * 3 + 0] += TN[i * 3 + 0];
			VN[v1 * 3 + 1] += TN[i * 3 + 1];
			VN[v1 * 3 + 2] += TN[i * 3 + 2];
			VN[v2 * 3 + 0] += TN[i * 3 + 0];
			VN[v2 * 3 + 1] += TN[i * 3 + 1];
			VN[v2 * 3 + 2] += TN[i * 3 + 2];
		}
		for (int t = 0; t<t_number; t++)
			Normalize(&TN[t * 3]);
		for (int i = 0; i<number; i++)
			Normalize(&VN[i * 3]);

		if(invertNormal) for (int i = 0; i < 3 * number; i++)
			VN[i] = -VN[i];
	}

	void Build_Connectivity()
	{
		Build_Edges();
		Build_Neighborhood();
		Build_VT_Num();
		Build_Boundary();
	}

	void Build_Edges()
	{
		if (E == NULL) E = new int[max_number * 3 * 2];
		if (ET == NULL) ET = new int[max_number * 3 * 2];
		if (TE == NULL) TE = new int[max_number * 2 * 3];

		//Build E, ET and TE
		e_number = 0;
		int *_RE = new int[t_number * 9];
		for (int t = 0; t<t_number; t++)
		{
			int v0 = T[t * 3 + 0];
			int v1 = T[t * 3 + 1];
			int v2 = T[t * 3 + 2];

			if (v0<v1) { _RE[t * 9 + 0] = v0; _RE[t * 9 + 1] = v1; _RE[t * 9 + 2] = t; }
			else { _RE[t * 9 + 0] = v1; _RE[t * 9 + 1] = v0; _RE[t * 9 + 2] = t; }
			if (v1<v2) { _RE[t * 9 + 3] = v1; _RE[t * 9 + 4] = v2; _RE[t * 9 + 5] = t; }
			else { _RE[t * 9 + 3] = v2; _RE[t * 9 + 4] = v1; _RE[t * 9 + 5] = t; }
			if (v2<v0) { _RE[t * 9 + 6] = v2; _RE[t * 9 + 7] = v0; _RE[t * 9 + 8] = t; }
			else { _RE[t * 9 + 6] = v0; _RE[t * 9 + 7] = v2; _RE[t * 9 + 8] = t; }
		}
		Quick_Sort_RE(_RE, 0, t_number * 3 - 1);

		for (int i = 0; i<t_number * 3; i++)
		{
			if (i != 0 && _RE[i * 3] == _RE[(i - 1) * 3] && _RE[i * 3 + 1] == _RE[(i - 1) * 3 + 1])
			{
				//Add the edge to ET
				ET[e_number * 2 - 2] = _RE[i * 3 + 2];
				ET[e_number * 2 - 1] = _RE[i * 3 - 1];

				//Add the edge to TE
				int v0 = T[_RE[i * 3 + 2] * 3 + 0];
				int v1 = T[_RE[i * 3 + 2] * 3 + 1];
				int v2 = T[_RE[i * 3 + 2] * 3 + 2];
				if (v0 == _RE[i * 3 + 0] && v1 == _RE[i * 3 + 1] || v1 == _RE[i * 3 + 0] && v0 == _RE[i * 3 + 1])	TE[_RE[i * 3 + 2] * 3 + 0] = e_number - 1;
				if (v1 == _RE[i * 3 + 0] && v2 == _RE[i * 3 + 1] || v2 == _RE[i * 3 + 0] && v1 == _RE[i * 3 + 1])	TE[_RE[i * 3 + 2] * 3 + 1] = e_number - 1;
				if (v2 == _RE[i * 3 + 0] && v0 == _RE[i * 3 + 1] || v0 == _RE[i * 3 + 0] && v2 == _RE[i * 3 + 1])	TE[_RE[i * 3 + 2] * 3 + 2] = e_number - 1;
			}
			else
			{
				//Add the edge to E
				E[e_number * 2 + 0] = _RE[i * 3 + 0];
				E[e_number * 2 + 1] = _RE[i * 3 + 1];
				//Add the edge to ET
				ET[e_number * 2 + 0] = _RE[i * 3 + 2];
				ET[e_number * 2 + 1] = -1;
				//Add the edge to  TE
				int v0 = T[_RE[i * 3 + 2] * 3 + 0];
				int v1 = T[_RE[i * 3 + 2] * 3 + 1];
				int v2 = T[_RE[i * 3 + 2] * 3 + 2];
				if (v0 == _RE[i * 3 + 0] && v1 == _RE[i * 3 + 1] || v1 == _RE[i * 3 + 0] && v0 == _RE[i * 3 + 1])	TE[_RE[i * 3 + 2] * 3 + 0] = e_number;
				if (v1 == _RE[i * 3 + 0] && v2 == _RE[i * 3 + 1] || v2 == _RE[i * 3 + 0] && v1 == _RE[i * 3 + 1])	TE[_RE[i * 3 + 2] * 3 + 1] = e_number;
				if (v2 == _RE[i * 3 + 0] && v0 == _RE[i * 3 + 1] || v0 == _RE[i * 3 + 0] && v2 == _RE[i * 3 + 1])	TE[_RE[i * 3 + 2] * 3 + 2] = e_number;
				e_number++;
			}
		}
		delete[]_RE;

		// printf("e_number = %d\n", e_number);
	}
	
	void Build_Neighborhood()
	{
		if (vv_num == NULL) vv_num = new int[max_number];
		if (VV == NULL) VV = new int[max_number * 3 * 2];
		if (VE == NULL) VE = new int[max_number * 3 * 2];
		if (TT == NULL)	TT = new int[max_number * 2 * 3];

		//First set vv_num
		memset(vv_num, 0, sizeof(int)*max_number);
		for (int i = 0; i<e_number; i++)
		{
			vv_num[E[i * 2 + 0]]++;
			vv_num[E[i * 2 + 1]]++;
		}
		for (int i = 1; i<number; i++)
			vv_num[i] += vv_num[i - 1];
		for (int i = number; i>0; i--)
			vv_num[i] = vv_num[i - 1];
		vv_num[0] = 0;

		//Then set VV and VE
		int *_vv_num = new int[max_number];
		memcpy(_vv_num, vv_num, sizeof(int)*max_number);
		for (int i = 0; i<e_number; i++)
		{
			VV[_vv_num[E[i * 2 + 0]]] = E[i * 2 + 1];
			VV[_vv_num[E[i * 2 + 1]]] = E[i * 2 + 0];
			VE[_vv_num[E[i * 2 + 0]]++] = i;
			VE[_vv_num[E[i * 2 + 1]]++] = i;
		}
		delete[]_vv_num;

		for (int t = 0; t<t_number; t++)
		{
			TT[t * 3 + 0] = ET[TE[t * 3 + 0] * 2 + 0] + ET[TE[t * 3 + 0] * 2 + 1] - t;
			TT[t * 3 + 1] = ET[TE[t * 3 + 1] * 2 + 0] + ET[TE[t * 3 + 1] * 2 + 1] - t;
			TT[t * 3 + 2] = ET[TE[t * 3 + 2] * 2 + 0] + ET[TE[t * 3 + 2] * 2 + 1] - t;
		}
	}

	void Build_VT_Num()
	{
		if(vt_num == NULL) vt_num = new int[max_number];

		memset(vt_num, 0, sizeof(int)*number);
		for (int t = 0; t<t_number; t++)
		{
			vt_num[T[t * 3 + 0]]++;
			vt_num[T[t * 3 + 1]]++;
			vt_num[T[t * 3 + 2]]++;
		}
	}

	void Build_Boundary()
	{
		if(boundary == NULL) boundary = new int[max_number];

		memset(boundary, 0, sizeof(int)*number);
		for (int e = 0; e<e_number; e++)
		{
			boundary[E[e * 2 + 0]]++;
			boundary[E[e * 2 + 1]]++;
		}
		for (int v = 0; v<number; v++)
		{
			if (vt_num[v] == boundary[v])
				boundary[v] = 0;
			else boundary[v] = 1;
		}
	}



	//utility functions
	void Lap_Smooth_Mesh()
	{
		TYPE* next_X = new TYPE[max_number * 3];
		int num_iteration = 2;
		float lap_damping = 0.1;
		
		for (int l = 0; l < num_iteration; l++)
		{
			for (int i = 0; i < number; i++)
			{
				next_X[i * 3 + 0] = 0;
				next_X[i * 3 + 1] = 0;
				next_X[i * 3 + 2] = 0;

				for (int index = vv_num[i]; index < vv_num[i + 1]; index++)
				{
					int j = VV[index];
					next_X[i * 3 + 0] += X[j * 3 + 0] - X[i * 3 + 0];
					next_X[i * 3 + 1] += X[j * 3 + 1] - X[i * 3 + 1];
					next_X[i * 3 + 2] += X[j * 3 + 2] - X[i * 3 + 2];
				}

				next_X[i * 3 + 0] = X[i * 3 + 0] + next_X[i * 3 + 0] * lap_damping;
				next_X[i * 3 + 1] = X[i * 3 + 1] + next_X[i * 3 + 1] * lap_damping;
				next_X[i * 3 + 2] = X[i * 3 + 2] + next_X[i * 3 + 2] * lap_damping;
			}
			
			memcpy(X, next_X, sizeof(TYPE)*number * 3);
		}

		delete[] next_X;
	}
	
	void Select(TYPE p[], TYPE q[], int& select_v)
	{
		TYPE dir[3];
		dir[0] = q[0] - p[0];
		dir[1] = q[1] - p[1];
		dir[2] = q[2] - p[2];
		Normalize(dir);

		TYPE min_t = MY_INFINITE;
		int	 select_t;
		for (int t = 0; t<t_number; t++)
		{
			TYPE _min_t = MY_INFINITE;
			if (Ray_Triangle_Intersection(&X[T[t * 3 + 0] * 3], &X[T[t * 3 + 1] * 3], &X[T[t * 3 + 2] * 3], p, dir, _min_t) && _min_t>0 && _min_t<min_t)
			{
				select_t = t;
				min_t = _min_t;
			}
		}

		if (min_t != MY_INFINITE)	//Selection made
		{
			TYPE r;
			TYPE d0 = Squared_VE_Distance(&X[T[select_t * 3 + 0] * 3], p, q, r);
			TYPE d1 = Squared_VE_Distance(&X[T[select_t * 3 + 1] * 3], p, q, r);
			TYPE d2 = Squared_VE_Distance(&X[T[select_t * 3 + 2] * 3], p, q, r);
			if (d0<d1 && d0<d2)	select_v = T[select_t * 3 + 0];
			else if (d1<d2)		select_v = T[select_t * 3 + 1];
			else				select_v = T[select_t * 3 + 2];
		}
	}

	void Read_OBJ(const char *filename)
	{
		number = 0;
		t_number = 0;
		int vertex_normal_number = 0;
		FILE *fp = 0;
		fp = fopen(filename, "r+");
		if (fp == 0) { printf("ERROR: cannot open %s\n", filename); getchar(); }

		while (feof(fp) == 0)
		{
			char token[1024];
			fscanf(fp, "%s", &token, 1024);
			if (token[0] == '#' && token[1] == '\0')
			{
				int c;
				while (feof(fp) == 0)
					if ((c = fgetc(fp)) == '\r' || c == '\n')	break;
			}
			else if (token[0] == 'v' && token[1] == '\0')	//vertex
			{
				//fscanf(fp, "%lf %lf %lf\n", &X[number * 3], &X[number * 3 + 1], &X[number * 3 + 2]);
				fscanf(fp, "%f %f %f\n", &X[number * 3], &X[number * 3 + 1], &X[number * 3 + 2]);
				X[number * 3 + 0] = X[number * 3 + 0];
				X[number * 3 + 1] = X[number * 3 + 1];
				X[number * 3 + 2] = X[number * 3 + 2];
				number++;
			}
			else if (token[0] == 'v' && token[1] == 't')
			{
				float temp[2];
				//fscanf(fp, "%lf %lf\n", &temp[0], &temp[1]);
				fscanf(fp, "%f %f\n", &temp[0], &temp[1]);
			}
			else if (token[0] == 'v' && token[1] == 'n')
			{
				//fscanf(fp, "%lf %lf %lf\n", &VN[0 * 3], &VN[0 * 3 + 1], &VN[0 * 3 + 2]);
				fscanf(fp, "%f %f %f\n", &VN[0 * 3], &VN[0 * 3 + 1], &VN[0 * 3 + 2]);
			}
			else if (token[0] == 'f' && token[1] == '\0')
			{
				int data[16];
				int data_number = 0;

				/*fscanf(fp, "%s", &token, 1024);
				sscanf(token, "%d", &data[0], 1024);
				fscanf(fp, "%s", &token, 1024);
				sscanf(token, "%d", &data[3], 1024);
				fscanf(fp, "%s", &token, 1024);
				sscanf(token, "%d", &data[6], 1024);*/

				fscanf(fp, "%d/%d/%d %d/%d/%d %d/%d/%d", &data[0], &data[1], &data[2], &data[3], &data[4], &data[5], &data[6], &data[7], &data[8]);
				data_number = 9;
				if (fgetc(fp) == ' ')
				{
					data_number = 12;
					fscanf(fp, " %d/%d/%d", &data[9], &data[10], &data[11]);
				}

				T[t_number * 3 + 0] = data[0] - 1;
				T[t_number * 3 + 1] = data[3] - 1;
				T[t_number * 3 + 2] = data[6] - 1;
				t_number++;
				if (data_number == 12)
				{

					T[t_number * 3 + 0] = data[0] - 1;
					T[t_number * 3 + 1] = data[6] - 1;
					T[t_number * 3 + 2] = data[9] - 1;
					t_number++;
				}
			}
		}
		fclose(fp);


		/*TYPE minx = MY_INFINITE, maxx = -MY_INFINITE;
		TYPE miny = MY_INFINITE, maxy = -MY_INFINITE;
		TYPE minz = MY_INFINITE, maxz = -MY_INFINITE;
		for (int i = 0; i < number; i++)
		{
			if (X[3 * i + 0] > maxx) maxx = X[3 * i + 0];
			if (X[3 * i + 0] < minx) minx = X[3 * i + 0];
			if (X[3 * i + 1] > maxy) maxy = X[3 * i + 1];
			if (X[3 * i + 1] < miny) miny = X[3 * i + 1];
			if (X[3 * i + 2] > maxz) maxz = X[3 * i + 2];
			if (X[3 * i + 2] < minz) minz = X[3 * i + 2];
		}*/

		/*printf("obj file information\n");
		printf("v_number =  %d\n", number);
		printf("t_number = %d\n", t_number);
		printf("bbox: min = [%f %f %f], max = [%f %f %f]\n", minx, miny, minz, maxx, maxy, maxz);*/

	}

	void Write_OBJ(const char *filename)
	{
		FILE *fp = fopen(filename, "w+");
		for (int v = 0; v<number; v++)
			fprintf(fp, "v %f %f %f\n", X[v * 3 + 0], X[v * 3 + 1], X[v * 3 + 2]);

		for (int t = 0; t<t_number; t++)
			fprintf(fp, "f %d %d %d\n", T[t * 3 + 0] + 1, T[t * 3 + 1] + 1, T[t * 3 + 2] + 1);

		fclose(fp);
	}

	void Read_Manta_Parts_TXT(const char *filename)
	{
		number = 0;
		FILE *fp = 0;
		fp = fopen(filename, "r+");
		if (fp == 0) { printf("ERROR: cannot open %s\n", filename); getchar(); }

		int temp_int[10];
		float temp_float[10];

		fscanf(fp, "%d, pdata: %d (%d,%d,%d) \n", &temp_int[0], &temp_int[1], &temp_int[2], &temp_int[3], &temp_int[4]);
		number = temp_int[0];

		for (int i = 0; i < number; i++)
		{
			fscanf(fp, "%d: [%f,%f,%f] , %d. [%f,%f,%f] \n", &temp_int[0], &temp_float[0], &temp_float[1], &temp_float[2], &temp_int[2], &temp_float[3], &temp_float[4], &temp_float[5]);
			if (temp_int[0] != i) { printf("Error: index not match %d != %d\n", i, temp_int[0]); getchar(); }
			X[i * 3 + 0] = temp_float[0];
			X[i * 3 + 1] = temp_float[1];
			X[i * 3 + 2] = temp_float[2];

			//printf("%f %f %f\n", X[i * 3 + 0], X[i * 3 + 1], X[i * 3 + 2]);
		}

		fclose(fp);
	}

	void Read_Manta_Mesh_OBJ(const char *filename)
	{
		FILE *fp = 0;
		fp = fopen(filename, "r+");
		if (fp == 0) { printf("ERROR: cannot open %s\n", filename); getchar(); }

		number = 0;
		t_number = 0;
		fscanf(fp, "o MantaMesh\n");
		while (feof(fp) == 0)
		{
			char token[1024];
			fscanf(fp, "%s", &token, 1024);
			if (token[0] == 'o' && token[1] == '\0')
			{
				int c;
				while (feof(fp) == 0)
					if ((c = fgetc(fp)) == '\r' || c == '\n')	break;
			}
			else if (token[0] == 'v' && token[1] == '\0')	//vertex
			{
				fscanf(fp, "%f %f %f\n", &X[number * 3], &X[number * 3 + 1], &X[number * 3 + 2]);
				X[number * 3 + 0] = X[number * 3 + 0];
				X[number * 3 + 1] = X[number * 3 + 1];
				X[number * 3 + 2] = X[number * 3 + 2];
				number++;
			}
			else if (token[0] == 'f' && token[1] == '\0')
			{
				int data[3];
				fscanf(fp, "%d %d %d \n", &data[0], &data[1], &data[2]);
 
				T[t_number * 3 + 0] = data[0] - 1;
				T[t_number * 3 + 1] = data[1] - 1;
				T[t_number * 3 + 2] = data[2] - 1;
				t_number++;
			}
		}
		fclose(fp);
	}
	
	void Read_Manta_Mesh_GZ(const char *filename)
	{
		gzFile gzf = gzopen(filename, "rb1"); // do some compression
		if (!gzf){ printf("ERROR: cannot open %s\n", filename); getchar(); }
			
		number = 0;
		t_number = 0;

		// read vertices
		gzread(gzf, &number, sizeof(int));
		for (int i = 0; i<number; i++) {		
			gzread(gzf, &X[i * 3 + 0], sizeof(float));
			gzread(gzf, &X[i * 3 + 1], sizeof(float));
			gzread(gzf, &X[i * 3 + 2], sizeof(float));
		}

		// normals
		int normal_num = 0;
		float pos[3];
		gzread(gzf, &normal_num, sizeof(int));
		for (int i = 0; i<normal_num; i++) {
			gzread(gzf, &pos[0], sizeof(float) * 3);
		}

		// read tris
 		gzread(gzf, &t_number, sizeof(int));
		for (int t = 0; t<t_number; t++) {
			gzread(gzf, &T[t * 3 + 0], sizeof(int));
			gzread(gzf, &T[t * 3 + 1], sizeof(int));
			gzread(gzf, &T[t * 3 + 2], sizeof(int));
		}
		// note - vortex sheet info ignored for now... (see writeBobj)
		gzclose(gzf);
	}

	void Write_Oriented_Points_xyz(const char *filename)
	{
		FILE *fp = 0;
		fp = fopen(filename, "w+");
		if (fp == 0) { printf("ERROR: cannot open %s\n", filename); getchar(); }

		for (int v = 0; v<number; v++)
			fprintf(fp, "%f %f %f %f %f %f\n", X[v * 3 + 0], X[v * 3 + 1], X[v * 3 + 2], VN[v * 3 + 0], VN[v * 3 + 1], VN[v * 3 + 2]);

		fclose(fp);
	}

	void Write_Points_xyz(const char *filename)
	{
		FILE *fp = 0;
		fp = fopen(filename, "w+");
		if (fp == 0) { printf("ERROR: cannot open %s\n", filename); getchar(); }

		for (int v = 0; v < number; v++)
			fprintf(fp, "%f %f %f\n", X[v * 3 + 0], X[v * 3 + 1], X[v * 3 + 2]);

		fclose(fp);
	}

	void Draw_Triangles(int normal_mode = 0)
	{
		glEnable(GL_LIGHTING);
		for (int i = 0; i<t_number; i++)
		{
			int v0 = T[i * 3 + 0];
			int v1 = T[i * 3 + 1];
			int v2 = T[i * 3 + 2];
			if (v0 == v1 || v0 == v2 || v1 == v2)	continue;
			glBegin(GL_TRIANGLES);
			glNormal3d(TN[i * 3], TN[i * 3 + 1], TN[i * 3 + 2]);
			if (normal_mode)	glNormal3d(VN[v0 * 3], VN[v0 * 3 + 1], VN[v0 * 3 + 2]);
			glVertex3d(X[v0 * 3], X[v0 * 3 + 1], X[v0 * 3 + 2]);
			if (normal_mode)	glNormal3d(VN[v1 * 3], VN[v1 * 3 + 1], VN[v1 * 3 + 2]);
			glVertex3d(X[v1 * 3], X[v1 * 3 + 1], X[v1 * 3 + 2]);
			if (normal_mode)	glNormal3d(VN[v2 * 3], VN[v2 * 3 + 1], VN[v2 * 3 + 2]);
			glVertex3d(X[v2 * 3], X[v2 * 3 + 1], X[v2 * 3 + 2]);
			glEnd();
		}
		glDisable(GL_LIGHTING);
	}

	void Draw_Edges()
	{
		glUseProgram(0);
		glDisable(GL_LIGHTING);
		glColor3f(0.2, 0.2, 0.2);
		float offset = -0.001f;

		for (int e = 0; e<e_number; e++)
		{
			glBegin(GL_LINES);
			glVertex3d(
				X[E[e * 2 + 0] * 3 + 0] + VN[E[e * 2 + 0] * 3 + 0] * offset,
				X[E[e * 2 + 0] * 3 + 1] + VN[E[e * 2 + 0] * 3 + 1] * offset,
				X[E[e * 2 + 0] * 3 + 2] + VN[E[e * 2 + 0] * 3 + 2] * offset);
			glVertex3d(
				X[E[e * 2 + 1] * 3 + 0] + VN[E[e * 2 + 1] * 3 + 0] * offset,
				X[E[e * 2 + 1] * 3 + 1] + VN[E[e * 2 + 1] * 3 + 1] * offset,
				X[E[e * 2 + 1] * 3 + 2] + VN[E[e * 2 + 1] * 3 + 2] * offset);
			glEnd();
		}

		if (0)
			for (int t = 0; t < t_number; t++) if (t == 9124)
			{
				int v0 = T[t * 3 + 0];
				int v1 = T[t * 3 + 1];
				int v2 = T[t * 3 + 2];

				glBegin(GL_LINE_LOOP);
				glVertex3d(
					X[v0 * 3 + 0] + VN[v0 * 3 + 0] * offset,
					X[v0 * 3 + 1] + VN[v0 * 3 + 1] * offset,
					X[v0 * 3 + 2] + VN[v0 * 3 + 2] * offset);
				glVertex3d(
					X[v1 * 3 + 0] + VN[v1 * 3 + 0] * offset,
					X[v1 * 3 + 1] + VN[v1 * 3 + 1] * offset,
					X[v1 * 3 + 2] + VN[v1 * 3 + 2] * offset);
				glVertex3d(
					X[v2 * 3 + 0] + VN[v2 * 3 + 0] * offset,
					X[v2 * 3 + 1] + VN[v2 * 3 + 1] * offset,
					X[v2 * 3 + 2] + VN[v2 * 3 + 2] * offset);
				glEnd();
			}
		
		glEnable(GL_LIGHTING);
	}

	void Draw_Vertices()
	{
		glUseProgram(0);
		glDisable(GL_LIGHTING);
		glColor3f(1, 0, 0);
		for (int v = 0; v<number; v++) //if(boundary[v])
		{
			glPushMatrix();
			glTranslatef(X[v * 3 + 0], X[v * 3 + 1], X[v * 3 + 2]);
			glutSolidSphere(0.01, 10, 10);
			glPopMatrix();
		}
		glEnable(GL_LIGHTING);
	}

	void Center(TYPE c[])
	{
		c[0] = c[1] = c[2] = 0;
		TYPE mass_sum = 0;
		for (int i = 0; i<number; i++)
		{
			c[0] += M[i] * X[i * 3 + 0];
			c[1] += M[i] * X[i * 3 + 1];
			c[2] += M[i] * X[i * 3 + 2];
			mass_sum += M[i];
		}
		c[0] /= mass_sum;
		c[1] /= mass_sum;
		c[2] /= mass_sum;
	}

	void Centralize()
	{
		TYPE c[3];
		Center(c);
		for (int i = 0; i<number; i++)
		{
			X[i * 3 + 0] -= c[0];
			X[i * 3 + 1] -= c[1];
			X[i * 3 + 2] -= c[2];
		}
	}

	//math functions
	TYPE Magnitude(TYPE *x)
	{
		return sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
	}

	TYPE Normalize(TYPE *x)
	{
		TYPE m = Magnitude(x);
		if (m<1e-14f)	
			return m;
		TYPE inv_m = 1 / m;
		x[0] *= inv_m;
		x[1] *= inv_m;
		x[2] *= inv_m;
		return m;
	}

	void Cross(TYPE* a, TYPE* b, TYPE* r)
	{
		r[0] = a[1] * b[2] - a[2] * b[1];
		r[1] = a[2] * b[0] - a[0] * b[2];
		r[2] = a[0] * b[1] - a[1] * b[0];
	}

	TYPE Dot(TYPE *v0, TYPE *v1)
	{
		return v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
	}

	bool Ray_Triangle_Intersection(TYPE x0[], TYPE x1[], TYPE x2[], TYPE p0[], TYPE dir[], TYPE &min_t)
	{
		TYPE e1[3], e2[3], s1[3];
		e1[0] = x1[0] - x0[0];
		e1[1] = x1[1] - x0[1];
		e1[2] = x1[2] - x0[2];
		e2[0] = x2[0] - x0[0];
		e2[1] = x2[1] - x0[1];
		e2[2] = x2[2] - x0[2];
		Cross(dir, e2, s1);
		TYPE divisor = Dot(s1, e1);
		if (divisor == 0) return false;
		// Test the first barycentric coordinate
		TYPE tt[3];
		tt[0] = p0[0] - x0[0];
		tt[1] = p0[1] - x0[1];
		tt[2] = p0[2] - x0[2];
		TYPE b1 = Dot(tt, s1);
		if (divisor>0 && (b1<0 || b1>divisor))		return false;
		if (divisor<0 && (b1>0 || b1<divisor))		return false;
		// Test the second barycentric coordinate
		TYPE s2[3];
		Cross(tt, e1, s2);
		TYPE b2 = Dot(dir, s2);
		if (divisor>0 && (b2<0 || b1 + b2>divisor))	return false;
		if (divisor<0 && (b2>0 || b1 + b2<divisor))	return false;
		// Compute t to intersection point
		min_t = Dot(e2, s2) / divisor;
		return min_t>0;
	}

	TYPE Squared_VE_Distance(const TYPE xi[], const TYPE xa[], const TYPE xb[], TYPE &r, TYPE *N = 0)
	{
		TYPE xia[3], xba[3];
		for (int n = 0; n<3; n++)
		{
			xia[n] = xi[n] - xa[n];
			xba[n] = xb[n] - xa[n];
		}
		TYPE xia_xba = Dot(xia, xba);
		TYPE xba_xba = Dot(xba, xba);
		if (xia_xba<0)				r = 0;
		else if (xia_xba>xba_xba)	r = 1;
		else						r = xia_xba / xba_xba;
		TYPE _N[3];
		if (N == 0)	N = _N;
		N[0] = xi[0] - xa[0] * (1 - r) - xb[0] * r;
		N[1] = xi[1] - xa[1] * (1 - r) - xb[1] * r;
		N[2] = xi[2] - xa[2] * (1 - r) - xb[2] * r;
		return Dot(N, N);
	}

	void Quick_Sort_RE(int a[], int l, int r)
	{
		if (l<r)
		{
			int j = Quick_Sort_Partition_RE(a, l, r);

			Quick_Sort_RE(a, l, j - 1);
			Quick_Sort_RE(a, j + 1, r);
		}
	}

	int Quick_Sort_Partition_RE(int a[], int l, int r)
	{
		int pivot[3], i, j, t[3];
		pivot[0] = a[l * 3 + 0];
		pivot[1] = a[l * 3 + 1];
		pivot[2] = a[l * 3 + 2];
		i = l; j = r + 1;
		while (1)
		{
			do ++i; while ((a[i * 3]<pivot[0] || a[i * 3] == pivot[0] && a[i * 3 + 1] <= pivot[1]) && i <= r);
			do --j; while (a[j * 3]>pivot[0] || a[j * 3] == pivot[0] && a[j * 3 + 1]> pivot[1]);
			if (i >= j) break;
			//Swap i and j			
			t[0] = a[i * 3 + 0];
			t[1] = a[i * 3 + 1];
			t[2] = a[i * 3 + 2];
			a[i * 3 + 0] = a[j * 3 + 0];
			a[i * 3 + 1] = a[j * 3 + 1];
			a[i * 3 + 2] = a[j * 3 + 2];
			a[j * 3 + 0] = t[0];
			a[j * 3 + 1] = t[1];
			a[j * 3 + 2] = t[2];
		}
		//Swap l and j
		t[0] = a[l * 3 + 0];
		t[1] = a[l * 3 + 1];
		t[2] = a[l * 3 + 2];
		a[l * 3 + 0] = a[j * 3 + 0];
		a[l * 3 + 1] = a[j * 3 + 1];
		a[l * 3 + 2] = a[j * 3 + 2];
		a[j * 3 + 0] = t[0];
		a[j * 3 + 1] = t[1];
		a[j * 3 + 2] = t[2];
		return j;
	}
};

#endif