#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <omp.h>
void mm( int m, int n, float *An, float *Cn);
void mm_32( int m, int n, float *An, float *Cn);
void p_tail_4( int m, int r, int a, int x, float *An, float *Cn);
void tail(int r, int x, int a, float *An, float *Cn);

void sgemm( int m, int n, float *A, float *C )
{	
	int largest = ((m > n) ? m : n);
	short use_mm = (largest % 20 < largest % 28);

	/* Matrix Multiply */
	if(use_mm)
		mm( m,n, A+0, C+0);
	else
	{
		mm_32(m,n,A+0,C+0);
	}
}

void mm( int r, int x, float *An, float *Cn)
{
	int BLOCKSIZE = 20;
	int m = ( r / BLOCKSIZE ) * BLOCKSIZE;
	int n =( x / BLOCKSIZE ) * BLOCKSIZE;
	int i,j,k, blockInd1;
	__m128 c0,c1,c2,c3,c4,a0,a1,a2,a3,a4,a0T;

/*-----------------------------------------------------------------------*/

	/* Start parallel multiply big block */
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
				/* Multiply */
				a0 = _mm_mul_ps(a0, a0T);
				a1 = _mm_mul_ps(a1, a0T);
				a2 = _mm_mul_ps(a2, a0T);
				a3 = _mm_mul_ps(a3, a0T);
				a4 = _mm_mul_ps(a4, a0T);
				/* Add  */
				c0 = _mm_add_ps(c0, a0);
				c1 = _mm_add_ps(c1, a1);
				c2 = _mm_add_ps(c2,a2);
				c3 = _mm_add_ps(c3,a3);
				c4 = _mm_add_ps(c4, a4);
			}

			/* Store the registers back into Cn */
			_mm_storeu_ps(Cn+blockInd1+j*r, c0);
			_mm_storeu_ps(Cn+blockInd1+4+j*r, c1);
			_mm_storeu_ps(Cn+blockInd1+8+j*r, c2);
			_mm_storeu_ps(Cn+blockInd1+12+j*r, c3);
			_mm_storeu_ps(Cn+blockInd1+16+j*r, c4);

		}			
	}

	/* End Parallel Multiply Big Block */

/*---------------------------------------------------------------*/

/* Start Parallel Tail */

/* METHOD 1 for Dealing with Parallel Tail*/
/*-----------------------------------------------------*/
	int a = m + (r-m)/4*4;
	int b = n + (x-n)/4*4;
	if(a != m || b !=n)
	{
		p_tail_4(m,r,a,x, An+0, Cn+0);
		tail(r,x,a,An+0,Cn+0);
	}

/* METHOD 2 for Dealing with Parallel Tail*/
/*-----------------------------------------------------*/
/*
	int a = m + (r-m)/16*16;
	int b = n + (x-n)/16*16;
	
	int c = m + (r-m)/12*12;
   int d = n + (x-n)/12*12;

	int e = m + (r-m)/8*8;
   int f = n + (x-n)/8*8;

	int g = m + (r-m)/4*4;
   int h = n + (x-n)/4*4;

	if(a != m || b != n)
	{
		p_tail_4(m,r,a-12,x, An+0, Cn+0);
		p_tail_4(m+4,r,a-8,x, An+0, Cn+0);
		p_tail_4(m+8,r,a-4,x, An+0, Cn+0);
		p_tail_4(m+12,r,a,x, An+0, Cn+0);
		
		tail(r,x,a,An+0,Cn+0);	
	}
	else if(c != m || d != n)
	{
	   p_tail_4(m,r,c-8,x, An+0, Cn+0);
		p_tail_4(m+4,r,c-4,x, An+0, Cn+0);
		p_tail_4(m+8,r,c,x, An+0, Cn+0);
		tail(r,x,c,An+0,Cn+0);	
	}
	else if(e != m || f != n)
   {
      p_tail_4(m,r,e-4,x, An+0, Cn+0);
      p_tail_4(m+4,r,e,x, An+0, Cn+0);
      tail(r,x,e,An+0,Cn+0);
   }
	else if(g != m || h != n)
   {
      p_tail_4(m,r,g,x, An+0, Cn+0);     
      tail(r,x,g,An+0,Cn+0);
   }
	else if( r % BLOCKSIZE != 0 || x % BLOCKSIZE != 0)
		tail(r,x,a,An+0,Cn+0);
*/
/*
		//parallel bottom
		for( int j = 0; j < a;j++)
		{
			for( int i = m; i < a; i+=4)
			{
				c0 = _mm_loadu_ps(Cn+i+j*r);
				for( int k = 0; k < x;k++)
				{
					a0T = _mm_load1_ps(An+j+k*r);

					a0 = _mm_loadu_ps(An+i+k*r);

					a0 = _mm_mul_ps(a0, a0T);
				
					c0 = _mm_add_ps(c0, a0);
				}
				_mm_storeu_ps(Cn+i+j*r, c0);
			}
		}

		//parallel sides
		for( int j = m; j < a; j++)
		{
			for( int i = 0; i < m; i+=4)
			{
				c0 = _mm_loadu_ps(Cn+i+j*r);
				for( int k = 0; k < x; k++)
				{
					a0T = _mm_load1_ps(An+j+k*r);

					a0 = _mm_loadu_ps(An+i+k*r);
		
					a0 = _mm_mul_ps(a0, a0T);

					c0 = _mm_add_ps(c0, a0);
				}
				_mm_storeu_ps(Cn+i+j*r, c0);
			}
		}
*/

//	}  DONT FORGET THAT THIS BRACKET BELONGS TO THE TOP MOST IF STATEMENT IF YOU DECIDE TO REVERT!!!

	//begin handle small tail
//	if( r % BLOCKSIZE != 0 || x % BLOCKSIZE != 0)
//	{
//		tail(r,x,a,An+0,Cn+0);
/*
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
*/
//	}
	//end handle tail cases

}
/* End Program */

void mm_32( int r, int x, float *An, float *Cn)
{
	int BLOCKSIZE = 28;
	int m = ( r / BLOCKSIZE ) * BLOCKSIZE;
	int n =( x / BLOCKSIZE ) * BLOCKSIZE;
	int i,j,k, blockInd1;
	__m128 c0,c1,c2,c3,c4,c5,c6,a0,a1,a2,a3,a4,a5,a6,a0T;

/*------------------------------------------------------------------------*/

	/* Start parallel multiply big block */	

	for( j = 0; j < m; j++)
	{
		for(blockInd1 = 0; blockInd1 < m; blockInd1 += BLOCKSIZE)
		{	
			/* Load C data into registers */
 			c0	= _mm_loadu_ps(Cn+blockInd1+j*r);
			c1 = _mm_loadu_ps(Cn+blockInd1+4+j*r);
			c2 = _mm_loadu_ps(Cn+blockInd1+8+j*r);
			c3 = _mm_loadu_ps(Cn+blockInd1+12+j*r);
			c4 = _mm_loadu_ps(Cn+blockInd1+16+j*r);

			/* Experiment */
			c5 = _mm_loadu_ps(Cn+blockInd1+20+j*r);
			c6 = _mm_loadu_ps(Cn+blockInd1+24+j*r);
//			c7 = _mm_loadu_ps(Cn+blockInd1+28+j*r);
//			c8 = _mm_loadu_ps(Cn+blockInd1+32+j*r);
//			c9 = _mm_loadu_ps(Cn+blockInd1+36+j*r);
//			c10 = _mm_loadu_ps(Cn+blockInd1+40+j*r);
//			c11 = _mm_loadu_ps(Cn+blockInd1+44+j*r);
//			c12 = _mm_loadu_ps(Cn+blockInd1+48+j*r);
//			c13 = _mm_loadu_ps(Cn+blockInd1+52+j*r);
					

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
/*
				a0 = _mm_loadu_ps(An+blockInd1+4+k*r);
				a0 = _mm_mul_ps(a0, a0T);
				c1 = _mm_add_ps(c1, a0);

				a0 = _mm_loadu_ps(An+blockInd1+8+k*r);
				a0 = _mm_mul_ps(a0, a0T);
				c2 = _mm_add_ps(c2, a0);

				a0 = _mm_loadu_ps(An+blockInd1+12+k*r);
				a0 = _mm_mul_ps(a0, a0T);
				c3 = _mm_add_ps(c3, a0);

				a0 = _mm_loadu_ps(An+blockInd1+16+k*r);
				a0 = _mm_mul_ps(a0, a0T);
				c4 = _mm_add_ps(c4, a0);

				a0 = _mm_loadu_ps(An+blockInd1+20+k*r);
				a0 = _mm_mul_ps(a0, a0T);
				c5 = _mm_add_ps(c5, a0);

				a0 = _mm_loadu_ps(An+blockInd1+24+k*r);
				a0 = _mm_mul_ps(a0, a0T);
				c6 = _mm_add_ps(c6, a0);
			
				a0 = _mm_loadu_ps(An+blockInd1+28+k*r);
				a0 = _mm_mul_ps(a0, a0T);
				c7 = _mm_add_ps(c7, a0);

				a0 = _mm_loadu_ps(An+blockInd1+32+k*r);
				a0 = _mm_mul_ps(a0, a0T);
				c8 = _mm_add_ps(c8, a0);

				a0 = _mm_loadu_ps(An+blockInd1+36+k*r);
				a0 = _mm_mul_ps(a0, a0T);
				c9 = _mm_add_ps(c9, a0);

				a0 = _mm_loadu_ps(An+blockInd1+40+k*r);
				a0 = _mm_mul_ps(a0, a0T);
				c10 = _mm_add_ps(c10, a0);

				a0 = _mm_loadu_ps(An+blockInd1+44+k*r);
				a0 = _mm_mul_ps(a0, a0T);
				c11 = _mm_add_ps(c11, a0);

				a0 = _mm_loadu_ps(An+blockInd1+48+k*r);
				a0 = _mm_mul_ps(a0, a0T);
				c12 = _mm_add_ps(c12, a0);

				a0 = _mm_loadu_ps(An+blockInd1+52+k*r);
				a0 = _mm_mul_ps(a0, a0T);
				c13 = _mm_add_ps(c13, a0);
*/

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
//			_mm_storeu_ps(Cn+blockInd1+28+j*r,c7);
//			_mm_storeu_ps(Cn+blockInd1+32+j*r,c8);
//			_mm_storeu_ps(Cn+blockInd1+36+j*r,c9);
//			_mm_storeu_ps(Cn+blockInd1+40+j*r,c10);
//			_mm_storeu_ps(Cn+blockInd1+44+j*r,c11);
//			_mm_storeu_ps(Cn+blockInd1+48+j*r,c12);
//			_mm_storeu_ps(Cn+blockInd1+52+j*r,c13);


		}			
	}

	/* End Parallel Multiply Big Block */

/*---------------------------------------------------------------------*/
   int a = m + (r-m)/16*16;
   int b = n + (x-n)/16*16;

   int c = m + (r-m)/12*12;
   int d = n + (x-n)/12*12;

   int e = m + (r-m)/8*8;
   int f = n + (x-n)/8*8;

   int g = m + (r-m)/4*4;
   int h = n + (x-n)/4*4;

   if(a != m || b != n)
   {
      p_tail_4(m,r,a-12,x, An+0, Cn+0);
      p_tail_4(m+4,r,a-8,x, An+0, Cn+0);
      p_tail_4(m+8,r,a-4,x, An+0, Cn+0);
      p_tail_4(m+12,r,a,x, An+0, Cn+0);
      tail(r,x,a,An+0,Cn+0);
   }
   else if(c != m || d != n)
   {
      p_tail_4(m,r,c-8,x, An+0, Cn+0);
      p_tail_4(m+4,r,c-4,x, An+0, Cn+0);
      p_tail_4(m+8,r,c,x, An+0, Cn+0);
      tail(r,x,c,An+0,Cn+0);
   }
   else if(e != m || f != n)
   {
      p_tail_4(m,r,e-4,x, An+0, Cn+0);
      p_tail_4(m+4,r,e,x, An+0, Cn+0);
      tail(r,x,e,An+0,Cn+0);
   }
   else if(g != m || h != n)
   {
      p_tail_4(m,r,g,x, An+0, Cn+0);
      tail(r,x,g,An+0,Cn+0);
   }
   else if( r % BLOCKSIZE != 0 || x % BLOCKSIZE != 0)
      tail(r,x,a,An+0,Cn+0);

	/* Start Parallel Tail */
/*---------------------------------------------------*/

/*
		//parallel bottom
		for( int j = 0; j < a;j++)
			for( int i = m; i < a; i+=4)
			{
				c0 = _mm_loadu_ps(Cn+i+j*r);
				for( int k = 0; k < x;k++)
				{
					a0T = _mm_load1_ps(An+j+k*r);

					a0 = _mm_loadu_ps(An+i+k*r);

					a0 = _mm_mul_ps(a0, a0T);
				
					c0 = _mm_add_ps(c0, a0);
				}
				_mm_storeu_ps(Cn+i+j*r, c0);
			}

		//parallel sides
		for( int j = m; j < a; j++)
			for( int i = 0; i < m; i+=4)
			{
				c0 = _mm_loadu_ps(Cn+i+j*r);
				for( int k = 0; k < x; k++)
				{
					a0T = _mm_load1_ps(An+j+k*r);

					a0 = _mm_loadu_ps(An+i+k*r);
		
					a0 = _mm_mul_ps(a0, a0T);

					c0 = _mm_add_ps(c0, a0);
				}
				_mm_storeu_ps(Cn+i+j*r, c0);
			}
*/
//	}
	/* End Parallel Tail */
/*----------------------------------------------------*/

	//begin handle small tail
//	if( r % BLOCKSIZE != 0 || x % BLOCKSIZE != 0)
//	{
//		tail(r,x,a, An+0, Cn+0);
/*
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
*/
//	}
	//end handle small tail

}
/* End Program */

void tail( int r, int x, int a, float *An, float *Cn)
{
	int i,j,k;			
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

void p_tail_4(int m, int r, int a, int x, float *An, float *Cn)
{
	int i,j,k;
	__m128 c0,a0,a0T;
		//parallel bottom
		for( j = 0; j < a;j++)
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

void p_tail_8(int m, int r, int a, int x, float *An, float *Cn)
{
	int i,j,k;
	__m128 c0,c1,a0,a1,a0T;
		//parallel bottom
		for( j = 0; j < a;j++)
		{
			for( i = m; i < a; i+=8 ) 
			{
				c0 = _mm_loadu_ps(Cn+i+j*r);
				c1 = _mm_loadu_ps(Cn+i+4+j*r);
				for( k = 0; k < x; k++ )
				{
					a0T = _mm_load1_ps(An+j+k*r);

					a0 = _mm_loadu_ps(An+i+k*r);
					a1 = _mm_loadu_ps(An+i+4+k*r);

					a0 = _mm_mul_ps(a0, a0T);
					a1 = _mm_mul_ps(a1, a0T);
				
					c0 = _mm_add_ps(c0, a0);
					c1 = _mm_add_ps(c1, a1);
				}
				_mm_storeu_ps(Cn+i+j*r, c0);
				_mm_storeu_ps(Cn+i+4+j*r, c1);
			}
		}

		//parallel sides
		for( j = m; j < a; j++)
		{
			for( i = 0; i < m; i+=8)
			{
				c0 = _mm_loadu_ps(Cn+i+j*r);
				c1 = _mm_loadu_ps(Cn+i+4+j*r);
				for( k = 0; k < x; k++)
				{
					a0T = _mm_load1_ps(An+j+k*r);

					a0 = _mm_loadu_ps(An+i+k*r);
					a1 = _mm_loadu_ps(An+i+4+k*r);
		
					a0 = _mm_mul_ps(a0, a0T);
					a1 = _mm_mul_ps(a1, a0T);

					c0 = _mm_add_ps(c0, a0);
					c1 = _mm_add_ps(c1, a1);
				}
				_mm_storeu_ps(Cn+i+j*r, c0);
				_mm_storeu_ps(Cn+i+4+j*r, c1);
			}
		}

}
