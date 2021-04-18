#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <math.h>
#include <float.h>
#include <string.h>

#define BLOCKSIZE 20
void mm( int m, int n, float *An, float *Cn);

void sgemm( int m, int n, float *A, float *C )
{
	/* Matrix Multiply */
	mm( m,n, A+0, C+0);
}

void mm( int r, int x, float *An, float *Cn)
{
	int m = ( r / BLOCKSIZE ) * BLOCKSIZE;
	int n =( x / BLOCKSIZE ) * BLOCKSIZE;
	int i,j,k, blockInd1;
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

	/* Extra registers to run an experiment */
	//__m128 c5;
	//__m128 a5;
	//__m128 c6;
	//__m128 a6;


/*--------------------------------------------------------------------------------------*/
	/* Start Parallel Multiply the big block of the 
		matrix that is a multiple of BLOCKSIZE */

	for( j = 0; j < m; j++)
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
		//c5 = _mm_loadu_ps(Cn+blockInd1+20+j*r);
		//c6 = _mm_loadu_ps(Cn+blockInd1+24+j*r);

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
			//a5 = _mm_loadu_ps(An+blockInd1+20+k*r);
			//a6 = _mm_loadu_ps(An+blockInd1+24+k*r);

			/* Multiply */
			a0 = _mm_mul_ps(a0, a0T);
			a1 = _mm_mul_ps(a1, a0T);
			a2 = _mm_mul_ps(a2, a0T);
			a3 = _mm_mul_ps(a3, a0T);
			a4 = _mm_mul_ps(a4, a0T);

			/* Experiment */
			//a5 = _mm_mul_ps(a5,a0T);
			//a6 = _mm_mul_ps(a6,a0T);

			/* Add  */
			c0 = _mm_add_ps(c0, a0);
			c1 = _mm_add_ps(c1, a1);
			c2 = _mm_add_ps(c2,a2);
			c3 = _mm_add_ps(c3,a3);
			c4 = _mm_add_ps(c4, a4);

			/* Experiment */
			//c5 = _mm_add_ps(c5,a5);
			//c6 = _mm_add_ps(c6,a6);
	
		}

		/* Store the registers back into Cn */
		_mm_storeu_ps(Cn+blockInd1+j*r, c0);
		_mm_storeu_ps(Cn+blockInd1+4+j*r, c1);
		_mm_storeu_ps(Cn+blockInd1+8+j*r, c2);
		_mm_storeu_ps(Cn+blockInd1+12+j*r, c3);
		_mm_storeu_ps(Cn+blockInd1+16+j*r, c4);

		/* Experiment */
		//_mm_storeu_ps(Cn+blockInd1+20+j*r,c5);
		//_mm_storeu_ps(Cn+blockInd1+24+j*r,c6);

	}			
	}

	/* End Parallel Multiply Big Block */

/*-------------------------------------------------------------------------------------*/

	/* Begin handle tail cases */
	if( r % BLOCKSIZE != 0 || x % BLOCKSIZE != 0)
	{

	/* Fill bottom of C outside of BLOCKSIZE */
	for( j = 0; j < r; j++)
	{
		for( k = 0; k < x; k++)
		{
			for( i = m; i < r/4*4; i+=4)
			{
				Cn[i+j*r] += An[i+k*r] * An[j+k*r];
				Cn[i+1+j*r] += An[i+1+k*r] * An[j+k*r];
                                Cn[i+2+j*r] += An[i+2+k*r] * An[j+k*r];
                                Cn[i+3+j*r] += An[i+3+k*r] * An[j+k*r];
				//Cn[i+4+j*r] += An[i+4+k*r] * An[j+k*r];
                                //Cn[i+5+j*r] += An[i+5+k*r] * An[j+k*r];
                                //Cn[i+6+j*r] += An[i+6+k*r] * An[j+k*r];
				//Cn[i+7+j*r] += An[i+7+k*r] * An[j+k*r];
			}
			for( i = r/4*4; i < r; i++) Cn[i+j*r] += An[i+k*r] * An[j+k*r];
		}
	}

	/* Fill right side of C outside of BLOCKSIZE */
	for( j = m; j < r; j++)
	{
		for( k = 0; k < x; k++)
		{
			for( i = 0; i < m/4*4; i+=4)
			{
				Cn[i+j*r] += An[i+k*r] * An[j+k*r];
				Cn[i+1+j*r] += An[i+1+k*r] * An[j+k*r];
                                Cn[i+2+j*r] += An[i+2+k*r] * An[j+k*r];
                                Cn[i+3+j*r] += An[i+3+k*r] * An[j+k*r];
				//Cn[i+4+j*r] += An[i+4+k*r] * An[j+k*r];
                                //Cn[i+5+j*r] += An[i+5+k*r] * An[j+k*r];
                                //Cn[i+6+j*r] += An[i+6+k*r] * An[j+k*r];
                                //Cn[i+7+j*r] += An[i+7+k*r] * An[j+k*r];
			}
			for( i = m/4*4; i < m; i++) Cn[i+j*r] += An[i+k*r] * An[j+k*r];
		}
	}

	/* End Handle Tail Cases */
	}

/* End Program */
}

