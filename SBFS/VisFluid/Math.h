#ifndef __MATH_H__
#define __MATH_H__

#define ONE_THIRD		0.33333333333333333333f
#define SWAP(X, Y)      {temp={X}; X=(Y); Y=(temp);}
#define CLAMP(a, l, h)  (((a)>(h))?(h):(((a)<(l))?(l):(a)))
#define SIGN(a)			((a)<0?-1:1)
#define	MY_MIN(a,b)		((a)<(b)?(a):(b))
#define	MY_MAX(a,b)		((a)>(b)?(a):(b))

//R=A*B
template <class T> FORCEINLINE
void Matrix_Product_3(T *A, T *B, T *R, bool sum = false)		
{
	if (sum == false)	memset(R, 0, sizeof(T) * 9);
	R[0] += A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
	R[1] += A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
	R[2] += A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
	R[3] += A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
	R[4] += A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
	R[5] += A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
	R[6] += A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
	R[7] += A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
	R[8] += A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
}

//R=A*B'
template <class T> FORCEINLINE
void Matrix_Product_T_3(T *A, T *B, T *R, bool sum = false)	
{
	if (sum == false)	memset(R, 0, sizeof(T) * 9);
	R[0] += A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
	R[1] += A[0] * B[3] + A[1] * B[4] + A[2] * B[5];
	R[2] += A[0] * B[6] + A[1] * B[7] + A[2] * B[8];
	R[3] += A[3] * B[0] + A[4] * B[1] + A[5] * B[2];
	R[4] += A[3] * B[3] + A[4] * B[4] + A[5] * B[5];
	R[5] += A[3] * B[6] + A[4] * B[7] + A[5] * B[8];
	R[6] += A[6] * B[0] + A[7] * B[1] + A[8] * B[2];
	R[7] += A[6] * B[3] + A[7] * B[4] + A[8] * B[5];
	R[8] += A[6] * B[6] + A[7] * B[7] + A[8] * B[8];
}

//R=A'*B
template <class T> FORCEINLINE
void Matrix_T_Product_3(T *A, T *B, T *R, bool sum = false)
{
	if (sum == false)	memset(R, 0, sizeof(T) * 9);
	R[0] += A[0] * B[0] + A[3] * B[3] + A[6] * B[6];
	R[1] += A[0] * B[1] + A[3] * B[4] + A[6] * B[7];
	R[2] += A[0] * B[2] + A[3] * B[5] + A[6] * B[8];
	R[3] += A[1] * B[0] + A[4] * B[3] + A[7] * B[6];
	R[4] += A[1] * B[1] + A[4] * B[4] + A[7] * B[7];
	R[5] += A[1] * B[2] + A[4] * B[5] + A[7] * B[8];
	R[6] += A[2] * B[0] + A[5] * B[3] + A[8] * B[6];
	R[7] += A[2] * B[1] + A[5] * B[4] + A[8] * B[7];
	R[8] += A[2] * B[2] + A[5] * B[5] + A[8] * B[8];
}

//R=A*B'
template <class T> FORCEINLINE
void Matrix_Product_T(T *A, T *B, T *R, int nx, int ny, int nz)	
{
	memset(R, 0, sizeof(T)*nx*nz);
	for (int i = 0; i<nx; i++)
		for (int j = 0; j<nz; j++)
			for (int k = 0; k<ny; k++)
				R[i*nz + j] += A[i*ny + k] * B[j*ny + k];
}


//r=A*x
template <class T> FORCEINLINE
void Matrix_Vector_Product_3(T *A, T *x, T *r, bool sum = false)	
{
	if (sum == false)	memset(r, 0, sizeof(T) * 3);
	r[0] += A[0] * x[0] + A[1] * x[1] + A[2] * x[2];
	r[1] += A[3] * x[0] + A[4] * x[1] + A[5] * x[2];
	r[2] += A[6] * x[0] + A[7] * x[1] + A[8] * x[2];
}

//r=A'*x
template <class T> FORCEINLINE
void Matrix_T_Vector_Product_3(T *A, T *x, T *r, bool sum = false)
{
	if (sum == false)	memset(r, 0, sizeof(T) * 3);
	r[0] += A[0] * x[0] + A[3] * x[1] + A[6] * x[2];
	r[1] += A[1] * x[0] + A[4] * x[1] + A[7] * x[2];
	r[2] += A[2] * x[0] + A[5] * x[1] + A[8] * x[2];
}

//R=inv(A)
template <class T> FORCEINLINE
T Matrix_Inverse_3(T *A, T *R)				
{
	R[0] = A[4] * A[8] - A[7] * A[5];
	R[1] = A[7] * A[2] - A[1] * A[8];
	R[2] = A[1] * A[5] - A[4] * A[2];
	R[3] = A[5] * A[6] - A[3] * A[8];
	R[4] = A[0] * A[8] - A[2] * A[6];
	R[5] = A[2] * A[3] - A[0] * A[5];
	R[6] = A[3] * A[7] - A[4] * A[6];
	R[7] = A[1] * A[6] - A[0] * A[7];
	R[8] = A[0] * A[4] - A[1] * A[3];
	T det = A[0] * R[0] + A[3] * R[1] + A[6] * R[2];
	T inv_det = 1 / det;
	for (int i = 0; i<9; i++)	R[i] *= inv_det;
	return det;
}

///////////////////////////////////////////////////////////////////////////////////////////
//  SVD function <from numerical recipes in C++>
//		Given a matrix a[1..m][1..n], this routine computes its singular value
//		decomposition, A = U.W.VT.  The matrix U replaces a on output.  The diagonal
//		matrix of singular values W is output as a vector w[1..n].  The matrix V (not
//		the transpose VT) is output as v[1..n][1..n].
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE> FORCEINLINE
TYPE pythag(TYPE a, TYPE b)
{
	TYPE at = fabs(a), bt = fabs(b), ct, result;
	if (at > bt) { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
	else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
	else result = 0.0;
	return(result);
}

template <class TYPE>
void SVD3(TYPE u[3][3], TYPE w[3], TYPE v[3][3])
{
	TYPE	anorm, c, f, g, h, s, scale;
	TYPE	x, y, z;
	TYPE	rv1[3];
	g = scale = anorm = 0.0; //Householder reduction to bidiagonal form.

	for (int i = 0; i<3; i++)
	{
		int l = i + 1;
		rv1[i] = scale*g;
		g = s = scale = 0.0;
		if (i<3)
		{
			for (int k = i; k<3; k++) scale += fabsf(u[k][i]);
			if (scale != 0)
			{
				for (int k = i; k<3; k++)
				{
					u[k][i] /= scale;
					s += u[k][i] * u[k][i];
				}
				f = u[i][i];
				g = -sqrtf(s)*SIGN(f);
				h = f*g - s;
				u[i][i] = f - g;
				for (int j = l; j<3; j++)
				{
					s = 0;
					for (int k = i; k<3; k++)	s += u[k][i] * u[k][j];
					f = s / h;
					for (int k = i; k<3; k++)	u[k][j] += f*u[k][i];
				}
				for (int k = i; k<3; k++)		u[k][i] *= scale;
			}
		}
		w[i] = scale*g;

		g = s = scale = 0.0;
		if (i <= 2 && i != 2)
		{
			for (int k = l; k<3; k++)	scale += fabsf(u[i][k]);
			if (scale != 0)
			{
				for (int k = l; k<3; k++)
				{
					u[i][k] /= scale;
					s += u[i][k] * u[i][k];
				}
				f = u[i][l];
				g = -sqrtf(s)*SIGN(f);
				h = f*g - s;
				u[i][l] = f - g;
				for (int k = l; k<3; k++) rv1[k] = u[i][k] / h;
				for (int j = l; j<3; j++)
				{
					s = 0;
					for (int k = l; k<3; k++)	s += u[j][k] * u[i][k];
					for (int k = l; k<3; k++)	u[j][k] += s*rv1[k];
				}
				for (int k = l; k<3; k++) u[i][k] *= scale;
			}
		}
		anorm = MY_MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}

	for (int i = 2, l; i >= 0; i--) //Accumulation of right-hand transformations.
	{
		if (i<2)
		{
			if (g != 0)
			{
				for (int j = l; j<3; j++) //Double division to avoid possible under				
					v[j][i] = (u[i][j] / u[i][l]) / g;
				for (int j = l; j<3; j++)
				{
					s = 0;
					for (int k = l; k<3; k++)	s += u[i][k] * v[k][j];
					for (int k = l; k<3; k++)	v[k][j] += s*v[k][i];
				}
			}
			for (int j = l; j<3; j++)	v[i][j] = v[j][i] = 0.0;
		}
		v[i][i] = 1.0;
		g = rv1[i];
		l = i;
	}

	for (int i = 2; i >= 0; i--) //Accumulation of left-hand transformations.
	{
		int l = i + 1;
		g = w[i];
		for (int j = l; j<3; j++) u[i][j] = 0;
		if (g != 0)
		{
			g = 1 / g;
			for (int j = l; j<3; j++)
			{
				s = 0;
				for (int k = l; k<3; k++)	s += u[k][i] * u[k][j];
				f = (s / u[i][i])*g;
				for (int k = i; k<3; k++)	u[k][j] += f*u[k][i];
			}
			for (int j = i; j<3; j++)		u[j][i] *= g;
		}
		else for (int j = i; j<3; j++)		u[j][i] = 0.0;
		u[i][i]++;
	}

	for (int k = 2; k >= 0; k--)				//Diagonalization of the bidiagonal form: Loop over
	{
		for (int its = 0; its<30; its++)	//singular values, and over allowed iterations.
		{
			bool flag = true;
			int  l;
			int	 nm;
			for (l = k; l >= 0; l--)			//Test for splitting.
			{
				nm = l - 1;
				if ((TYPE)(fabs(rv1[l]) + anorm) == anorm)
				{
					flag = false;
					break;
				}
				if ((TYPE)(fabs(w[nm]) + anorm) == anorm)	break;
			}
			if (flag)
			{
				c = 0.0; //Cancellation of rv1[l], if l > 0.
				s = 1.0;
				for (int i = l; i<k + 1; i++)
				{
					f = s*rv1[i];
					rv1[i] = c*rv1[i];
					if ((TYPE)(fabs(f) + anorm) == anorm) break;
					g = w[i];
					h = pythag(f, g);
					w[i] = h;
					h = 1.0 / h;
					c = g*h;
					s = -f*h;
					for (int j = 0; j<3; j++)
					{
						y = u[j][nm];
						z = u[j][i];
						u[j][nm] = y*c + z*s;
						u[j][i] = z*c - y*s;
					}
				}
			}
			z = w[k];
			if (l == k)		// Convergence.
			{
				if (z<0.0)	// Singular value is made nonnegative.
				{
					w[k] = -z;
					for (int j = 0; j<3; j++) v[j][k] = -v[j][k];
				}
				break;
			}
			if (its == 29) { printf("Error: no convergence in 30 svdcmp iterations"); getchar(); }
			x = w[l]; //Shift from bottom 2-by-2 minor.
			nm = k - 1;
			y = w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z)*(y + z) + (g - h)*(g + h)) / (2.0*h*y);
			g = pythag(f, (TYPE)1.0);
			f = ((x - z)*(x + z) + h*((y / (f + fabs(g)*SIGN(f))) - h)) / x;
			c = s = 1.0; //Next QR transformation:

			for (int j = l; j <= nm; j++)
			{
				int i = j + 1;
				g = rv1[i];
				y = w[i];
				h = s*g;
				g = c*g;
				z = pythag(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x*c + g*s;
				g = g*c - x*s;
				h = y*s;
				y *= c;
				for (int jj = 0; jj<3; jj++)
				{
					x = v[jj][j];
					z = v[jj][i];
					v[jj][j] = x*c + z*s;
					v[jj][i] = z*c - x*s;
				}
				z = pythag(f, h);
				w[j] = z; //Rotation can be arbitrary if z D 0.
				if (z)
				{
					z = 1.0 / z;
					c = f*z;
					s = h*z;
				}
				f = c*g + s*y;
				x = c*y - s*g;
				for (int jj = 0; jj<3; jj++)
				{
					y = u[jj][j];
					z = u[jj][i];
					u[jj][j] = y*c + z*s;
					u[jj][i] = z*c - y*s;
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = x;
		}
	}

	//Make sure the two matrices are rotational
	int small_id;
	if (fabsf(w[0])<fabsf(w[1]) && fabsf(w[0])<fabsf(w[2]))	small_id = 0;
	else if (fabsf(w[1])<fabsf(w[2]))						small_id = 1;
	else													small_id = 2;
	if (u[0][0] * (u[1][1] * u[2][2] - u[1][2] * u[2][1]) + u[1][0] * (u[2][1] * u[0][2] - u[0][1] * u[2][2]) + u[2][0] * (u[0][1] * u[1][2] - u[1][1] * u[0][2])<0)
	{
		u[0][small_id] = -u[0][small_id];
		u[1][small_id] = -u[1][small_id];
		u[2][small_id] = -u[2][small_id];
		w[small_id] = -w[small_id];
	}
	if (v[0][0] * (v[1][1] * v[2][2] - v[1][2] * v[2][1]) + v[1][0] * (v[2][1] * v[0][2] - v[0][1] * v[2][2]) + v[2][0] * (v[0][1] * v[1][2] - v[1][1] * v[0][2])<0)
	{
		v[0][small_id] = -v[0][small_id];
		v[1][small_id] = -v[1][small_id];
		v[2][small_id] = -v[2][small_id];
		w[small_id] = -w[small_id];
	}
}

#endif