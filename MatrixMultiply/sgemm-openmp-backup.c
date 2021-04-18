#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <omp.h>
#define NUM_THREADS 16
void mm_28( int m, int n, int ind1, int ind2, float *An, float *Cn);
void p_tail_4( int m, int r, int a, int x, float *An, float *Cn);
void tail(int r, int x, int a, float *An, float *Cn);

void sgemm( int m, int n, float *A, float *C )
{	
	int BLOCKSIZE = 28;
  	int r = ( m / BLOCKSIZE ) * BLOCKSIZE;
  	int x = ( n / BLOCKSIZE ) * BLOCKSIZE;
 
   int a = r + (m-r)/4*4;
  	int b = x + (n-x)/4*4;
		
	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel	
	{
		int id = omp_get_thread_num();	
		for(int j = 0; j < 14; j++)
			if(id == j)
				mm_28(m,n,j*r/14,(j+1)*r/14,A+0,C+0);

   	if(a != r || b != x)
   	{
      	p_tail_4(r,m,a,n, A+0, C+0);
			if(m % BLOCKSIZE !=0 || n % BLOCKSIZE != 0)
      			tail(m,n,a,A+0,C+0);
  		}
	}
}

void mm_28( int r, int x, int ind1, int ind2, float *An, float *Cn)
{
	int BLOCKSIZE = 28;
	int m = ( r / BLOCKSIZE ) * BLOCKSIZE;
	int i,j,k, blockInd1;
	__m128 c0,c1,c2,c3,c4,c5,c6,a0,a1,a2,a3,a4,a5,a6,a0T;

/*------------------------------------------------------------------------*/
	#pragma omp nowait
	{
	/* Start parallel multiply big block */
	#pragma omp private(ind1,ind2,j,k,c0,c1,c2,c3,c4,c5,c6,a0,a1,a2,a3,a4,a5,a6,a0T,blockInd1)	
	for( j = ind1; j < ind2; j++)
	{
		for(blockInd1 = 0; blockInd1 < m; blockInd1 += BLOCKSIZE)
		{	
			/* Load C data into registers */
			c0 = _mm_loadu_ps(Cn+blockInd1+j*r);
			c1 = _mm_loadu_ps(Cn+blockInd1+4+j*r);
			c2 = _mm_loadu_ps(Cn+blockInd1+8+j*r);
			c3 = _mm_loadu_ps(Cn+blockInd1+12+j*r);
			c4 = _mm_loadu_ps(Cn+blockInd1+16+j*r);
			/* Experiment */
			c5 = _mm_loadu_ps(Cn+blockInd1+20+j*r);
			c6 = _mm_loadu_ps(Cn+blockInd1+24+j*r);
			//c7 = _mm_loadu_ps(Cn+blockInd1+28+j*r);
			for(k = 0; k < x; k++)
			{
				/* Load the value that will be multiplied across multiple a's */
				a0T = _mm_load1_ps(An+j+k*r);
				/* Load the values to be multiplied */
		 		a0 = _mm_loadu_ps(An+blockInd1+k*r);
		      a1 = _mm_loadu_ps(An+blockInd1+4+k*r);
				a2 = _mm_loadu_ps(An+blockInd1+8+k*r);
				a3 = _mm_loadu_ps(An+blockInd1+12+k*r);
				a4 = _mm_loadu_ps(An+blockInd1+16+k*r);
				/* Experiment */
				a5 = _mm_loadu_ps(An+blockInd1+20+k*r);
				a6 = _mm_loadu_ps(An+blockInd1+24+k*r);
				//a7 = _mm_loadu_ps(An+blockInd1+28+k*r);
				/* Multiply */
				a0 = _mm_mul_ps(a0, a0T);
				a1 = _mm_mul_ps(a1, a0T);
				a2 = _mm_mul_ps(a2, a0T);
				a3 = _mm_mul_ps(a3, a0T);
				a4 = _mm_mul_ps(a4, a0T);
				/* Experiment */
				a5 = _mm_mul_ps(a5,a0T);
				a6 = _mm_mul_ps(a6,a0T);
				//a7 = _mm_mul_ps(a7,a0T);
				/* Add  */
				c0 = _mm_add_ps(c0, a0);
				c1 = _mm_add_ps(c1, a1);
				c2 = _mm_add_ps(c2,a2);
				c3 = _mm_add_ps(c3,a3);
				c4 = _mm_add_ps(c4, a4);
				/* Experiment */
				c5 = _mm_add_ps(c5,a5);
				c6 = _mm_add_ps(c6,a6);
				//c7 = _mm_add_ps(c7,a7);
			}
			/* Store the registers back into Cn */
			_mm_storeu_ps(Cn+blockInd1+j*r, c0);
			_mm_storeu_ps(Cn+blockInd1+4+j*r, c1);
			_mm_storeu_ps(Cn+blockInd1+8+j*r, c2);
			_mm_storeu_ps(Cn+blockInd1+12+j*r, c3);
			_mm_storeu_ps(Cn+blockInd1+16+j*r, c4);
			/* Experiment */
			_mm_storeu_ps(Cn+blockInd1+20+j*r,c5);
			_mm_storeu_ps(Cn+blockInd1+24+j*r,c6);
			//_mm_storeu_ps(Cn+blockInd1+28+j*r,c7);
		}			
	}
		/* End Parallel Multiply Big Block */
	}
/*---------------------------------------------------------------------*/
}
/* End Program */

void tail( int r, int x, int a, float *An, float *Cn)
{

	#pragma omp nowait
	{
	int i,j,k, id = omp_get_thread_num();
	if(id == 14)
	{
		#pragma omp private(i,k,j)
		//small bottom
		for( j = 0; j < r; j++)
		{
			for( k = 0; k < x; k++)
			{
				for( i = a; i < r/2*2; i+=2)
				{
					Cn[i+j*r] += An[i+k*r] * An[j+k*r];
					Cn[i+1+j*r] += An[i+1+k*r] * An[j+k*r]; 
				}
				for( i = r/2*2; i < r; i++) Cn[i+j*r] += An[i+k*r] * An[j+k*r];
			}
		}	

		//small side
		for( j = a; j < r; j++)
		{
			for( k = 0; k < x; k++)
			{
				for( i = 0; i < a/2*2; i+=2)
				{
					Cn[i+j*r] += An[i+k*r] * An[j+k*r];
					Cn[i+1+j*r] += An[i+1+k*r] * An[j+k*r];       
				}	
				for( i = a/2*2; i < a; i++) Cn[i+j*r] += An[i+k*r] * An[j+k*r];
			}
		}
	}
	}
}

void p_tail_4(int m, int r, int a, int x, float *An, float *Cn)
{
	#pragma omp nowait
	{
	int i,j,k, id = omp_get_thread_num();
	__m128 c0,a0,a0T;

	if(id == 15)
	{
		#pragma omp private(i,j,k,c0,a0,a0T)
		//parallel bottom
		for( j = 0 ; j < a; j++)
		{
			for( i = m; i < a; i+=4 ) 
			{
				c0 = _mm_loadu_ps(Cn+i+j*r);
				for( k = 0; k < x; k++ )
				{
					a0T = _mm_load1_ps(An+j+k*r);

					a0 = _mm_loadu_ps(An+i+k*r);

					a0 = _mm_mul_ps(a0, a0T);
				
					c0 = _mm_add_ps(c0, a0);
				}
				_mm_storeu_ps(Cn+i+j*r, c0);
			}
		}
		#pragma omp private(i,j,k,c0,a0,a0T)
		//parallel sides
		for( j = m; j < a; j++)
		{
			for( i = 0; i < m; i+=4)
			{
				c0 = _mm_loadu_ps(Cn+i+j*r);
				for( k = 0; k < x; k++)
				{
					a0T = _mm_load1_ps(An+j+k*r);

					a0 = _mm_loadu_ps(An+i+k*r);
		
					a0 = _mm_mul_ps(a0, a0T);

					c0 = _mm_add_ps(c0, a0);
				}
				_mm_storeu_ps(Cn+i+j*r, c0);
			}
		}
	}
	}
}


