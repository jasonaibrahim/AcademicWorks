#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h>

#define BLOCKSIZE 16

void naive( int m, int n, float *A, float *C);
float *matrix_pad( int *m, int *n, float *A, int *size);
float *matrix_unpad( int *m, float *A);
float *Strasson( float *A, int size);
void transpose( int *blockInd1, int *blockInd2, int *size, float *dst, float *src);
void sum( float *src1, float *src2, float *dst);
void subtract( float *src1, float *src2, float *dst);
void mat_mul( float *src1, float *src2, float *dst);

void mat_mul( float *src1, float *src2, float *dst)
{
	int i,j,k;
	for( j = 0; j < BLOCKSIZE; j++)
		for(k = 0; k < BLOCKSIZE; k++)
			for( i = 0; i < BLOCKSIZE; i++)
				dst[i+j*BLOCKSIZE] = src1[i+k*BLOCKSIZE] * src2[k+j*BLOCKSIZE];
}

/* This function will tranpose a block */
void transpose( int *blockInd1, int *blockInd2, int *size, float *dst, float *src)
{
	int i,j;
	for( i = *blockInd1; i < *(blockInd1) + BLOCKSIZE; i++)
		for( j = *blockInd2; j < *(blockInd2) + BLOCKSIZE; j++)
			dst[j+i*(*size)] = src[i+j*(*size)];
}

void sum( float *src1, float *src2, float *dst)
{
	int i,j;
	for( i= 0; i < BLOCKSIZE; i++)
		for( j = 0; j < BLOCKSIZE; j++)
			dst[i+j*BLOCKSIZE] = src1[i+j*BLOCKSIZE] + src2[i+j*BLOCKSIZE];
}

void subtract( float *src1, float *src2, float *dst)
{
        int i,j;
        for( i= 0; i < BLOCKSIZE; i++)
                for( j = 0; j < BLOCKSIZE; j++)
                        dst[i+j*BLOCKSIZE] = src1[i+j*BLOCKSIZE] - src2[i+j*BLOCKSIZE];
}

float *Strasson( float *A, int size)
{
	int blockInd1, blockInd2;
	
	float *Cn = (float*) malloc(size*size*sizeof(float));
	/* Declare and allocate all memory for BLOCKSIZE x BLOCKSIZE matrices */
	float *a11 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *a12 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *a21 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *a22 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));

	float *b11 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *b12 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *b21 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *b22 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));

	float *a11pa22 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *b11pb22 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *a21pa22 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *b12mb22 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *b21mb11 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *a11pa12 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *a21ma11 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *b11pb12 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *a12ma22 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *b21pb22 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));

	float *M1 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *M2 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *M3 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *M4 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *M5 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *M6 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *M7 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));

	float *c11 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *c12 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *c21 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *c22 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	
	float *inter1 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));
	float *inter2 = (float*) malloc((BLOCKSIZE*BLOCKSIZE)*sizeof(float));

	for( blockInd1 = 0; blockInd1 < size; blockInd1 += (2*BLOCKSIZE))
		for( blockInd2 = 0; blockInd2 < size; blockInd2 += (2*BLOCKSIZE))
		{
			a11[blockInd1 + blockInd2 * size] = A[blockInd1 + blockInd2 * size];
			a12[blockInd1 + blockInd2 * size] = A[blockInd1 + (blockInd2+BLOCKSIZE) * size];
			a21[blockInd1 + blockInd2 * size] = A[(blockInd1+BLOCKSIZE) + blockInd2 * size];
			a22[blockInd1 + blockInd2 * size] = A[(blockInd1+BLOCKSIZE) + (blockInd2+BLOCKSIZE) * size];

			transpose(&blockInd1, &blockInd2, &size, b11+0,a11+0);
			transpose(&blockInd1, &blockInd2, &size, b12+0,a12+0);
			transpose(&blockInd1, &blockInd2, &size, b21+0,a21+0);
			transpose(&blockInd1, &blockInd2, &size, b22+0,a22+0);

			sum(a11+0,a22+0,a11pa22+0);
			sum(b11+0,b22+0,b11pb22+0);
			sum(a21+0,a22+0,a21pa22+0);
			subtract(b12+0,b22+0,b12mb22+0);
			subtract(b21+0,b11+0,b21mb11+0);
			sum(a11+0,a12+0,a11pa12+0);
			subtract(a21+0,a11+0,a21ma11+0);
			sum(b11+0,b12+0,b11pb12+0);
			subtract(a12+0,a22+0,a12ma22+0);
			sum(b21+0,b22+0,b21pb22+0);
	
			mat_mul( a11pa22+0, b11pb22+0, M1);
			mat_mul( a21pa22+0, b11+0, M2);
			mat_mul( a11+0, b12mb22+0, M3);
			mat_mul( a22+0, b21mb11+0, M4);
			mat_mul( a11pa12+0, b22+0, M5);
			mat_mul( a21ma11+0, b11pb12+0, M6);
			mat_mul( a12ma22+0, b11pb22+0, M7);

			sum( M1+0, M4+0, inter1+0);
			sum( M5+0, M7+0, inter2+0);
			sum( inter1+0, inter2+0, c11+0);

			sum( M3+0, M5+0, c12+0);

			sum( M2+0, M4+0, c21+0);

			subtract( M1+0, M2+0, inter1+0);
                        sum( M3+0, M6+0, inter2+0);
                        sum( inter1+0, inter2+0, c22+0);
			
			Cn[blockInd1 + blockInd2 * size] = c11[blockInd1 + blockInd2 * size];
                        Cn[blockInd1 + (blockInd2+BLOCKSIZE) * size] = c12[blockInd1 + blockInd2 * size];
                        Cn[(blockInd1+BLOCKSIZE) + blockInd2 * size] = c21[blockInd1 + blockInd2 * size];
                        Cn[(blockInd1+BLOCKSIZE) + (blockInd2+BLOCKSIZE) * size] = c22[blockInd1 + blockInd2 * size];
			
		}
	
	return Cn+0;
}

float *matrix_unpad( int *m, float *A)
{
	int i,j;
	float *o = malloc(((*m)*(*m))*sizeof(float));
	for(i = 0; i < *m; i++)
		for(j = 0; j < *m; j++)
			o[i+j*(*m)] = A[i+j*(*m)];
	free(A);
	A = NULL;
	return o;
			
}

float *matrix_pad( int *m, int *n, float *A, int *size)
{
	int i,j, largest = ((*m > *n) ? *m : *n);
	float *v;

	if (largest > 32 && largest < 64)
		*(size) = 64;
	else if (largest > 64 && largest < 128)
		*(size) = 128;
	else if (largest > 128 && largest < 256)
		*(size) = 256;	
	else if (largest > 256)
		*(size) = 512;	
	else
		*(size) = largest;

	v = malloc(((*size)*(*size))*sizeof(float));

	for(i = 0; i < (*n); i++)
		for(j = 0; j < (*m); j++)
			v[i+j*(*n)] = A[i+j*(*n)];

	for( i = (*m)*(*n); i < (*size)*(*size); i++)
		v[i] = 0; 
	return v;
}

void sgemm( int m, int n, float *A, float *C)
{
	int size;
	A = matrix_pad(&m, &n, A+0, &size);
	
	float *Cn = Strasson(A+0, size);
	//naive(m,n,A+0,C+0);
	C = matrix_unpad(&m, Cn+0);
}

void naive( int m, int n, float *A, float *C)
{
	for(int j = 0; j < m; j++)
		for(int k = 0; k < n; k++)
			for( int i = 0; i < m; i++)
				C[i+j*m] += A[i+k*m] * A[j+k*m];
}
