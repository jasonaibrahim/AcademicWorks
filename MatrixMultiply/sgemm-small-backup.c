#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <math.h>
#include <float.h>
#include <string.h>

#define BLOCKSIZE 20

float *matrix_pad(int m, int n, int z, float *A);
void matrix_unpad(int m, int z, float *Cn, float *C);
void naive( int m, int n, float *A,float *C);
void mm( int n, float *An, float *Cn);

void sgemm( int m, int n, float *A, float *C )
{
  	int z, largest = (( m > n) ? m : n);
  	z = (largest+BLOCKSIZE)/BLOCKSIZE*BLOCKSIZE;
	
	float *An;
	float *Cn;
	if(m == n && m % BLOCKSIZE ==0 && n % BLOCKSIZE ==0)
	{
		An = A;
		mm( m, A+0, C+0);
	}
	else
	{		
  		An = matrix_pad(m, n, z, A+0);
		//float *Cn = (float*) malloc(z*z*sizeof(float));
		Cn = matrix_pad(m,m,z,C+0);
		mm( z, An+0, Cn+0);
		matrix_unpad(m,z,Cn+0, C+0);
	}

}

void mm( int n, float *An, float *Cn)
{
	int i,j,k, blockInd1, blockInd2;
	__m128 c0;
        __m128 c1;
	__m128 c2;
	__m128 c3;
	__m128 c4;
	__m128 a0;
	__m128 a1;
	__m128 a2;
	__m128 a3;
	__m128 a4;
	__m128 a0T;


	
	for( k = 0; k < n; k++)
	{
	
	//for(blockInd1 = 0; blockInd1 < n; blockInd1 += BLOCKSIZE)

		for(j = 0; j < n; j++)
		{
			a0T = _mm_load1_ps(An+j+k*n);
			for(i = 0; i < n; i+=BLOCKSIZE)
			{
				a0 = _mm_loadu_ps(An+(i+k*n));

				a1 = _mm_loadu_ps(An+(i+4+k*n));
				a0 = _mm_mul_ps(a0, a0T);

				a2 = _mm_loadu_ps(An+(i+8+k*n));
				a1 = _mm_mul_ps(a1, a0T);

				a3 = _mm_loadu_ps(An+(i+12+k*n));
				a2 = _mm_mul_ps(a2,a0T);

				a4 = _mm_loadu_ps(An+(i+16+k*n));
				a3 = _mm_mul_ps(a3, a0T);

				c0 = _mm_loadu_ps(Cn+i+j*n);
				a4 = _mm_mul_ps(a4, a0T);

				c1 = _mm_loadu_ps(Cn+i+4+j*n);
				c0 = _mm_add_ps(c0, a0);

				c2 = _mm_loadu_ps(Cn+i+8+j*n);
				c1 = _mm_add_ps(c1, a1);

				c3 = _mm_loadu_ps(Cn+i+12+j*n);
				c2 = _mm_add_ps(c2,a2);

				c4 = _mm_loadu_ps(Cn+i+16+j*n);
				c3 = _mm_add_ps(c3,a3);

				_mm_storeu_ps(Cn+i+j*n,c0);
				c4 = _mm_add_ps(c4, a4);

				_mm_storeu_ps(Cn+4+i+j*n,c1);

				_mm_storeu_ps(Cn+8+i+j*n,c2);
	
				_mm_storeu_ps(Cn+12+i+j*n,c3);

				_mm_storeu_ps(Cn+16+i+j*n,c4);
			}
		}
		
	}			
		
}

void naive( int m, int n, float *A, float *C)
{
	for( int j = 0; j < m; j++)
		for(int k = 0; k < n; k++)
			for(int i = 0; i < m; i++)
				C[i+j*m] += A[i+k*m] * A[j+k*m];
}

float *matrix_pad(int m, int n, int z, float *A)
{
   	int i,j;
   	float *temp = (float*) malloc( z * z * sizeof(float));
	float zero[4] = {0.0,0.0,0.0,0.0};;
	__m128 reg1;
	__m128 reg2;
	__m128 reg3;
	__m128 reg4;
	__m128 reg5;

   	for( i = 0; i < ((z)*(z))/20*20; i+=20) 
   	{
		reg1 = _mm_load1_ps(zero);
		_mm_storeu_ps(temp+i+0, reg1);
		reg2 = _mm_load1_ps(zero);
		_mm_storeu_ps(temp+i+4, reg2);
		reg3 = _mm_load1_ps(zero);
		_mm_storeu_ps(temp+i+8, reg3);
		reg4 = _mm_load1_ps(zero);
		_mm_storeu_ps(temp+i+12, reg4);
		reg5 = _mm_load1_ps(zero);
		_mm_storeu_ps(temp+i+16, reg5);
   	}
        for(i = ((z)*(z))/20*20; i < ((z)*(z)); i++)
       	     temp[i] = 0;

   for(i=0; i < n; i++)
   {
   	for(j = 0; j < m; j++)
	{
		temp[j+i*(z)] = A[j+i*(m)];
	}
   }
   return temp;   
}

void matrix_unpad( int m, int z, float *Cn, float *C)
{   
    int i,j;
   
    for(i = 0; i < m; i++)
    {
	for(j = 0; j < m; j++)
	{
		C[j+i*m] = Cn[j+i*z];
	}
    }

    return;
}

