// System includes
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>
// CUDA runtime
#include <cuda_runtime.h>
// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"

#ifndef BLOCKSIZE
#define BLOCKSIZE		32     // number of threads per block
#endif

const int Size = 32769;

//Function Prototypes
double square (int);

__global__ void AutoCorrelate(float *dA, float *dSum){
    int gid  = blockIdx.x*blockDim.x + threadIdx.x;
    int shift = gid;
    float sum = 0.;

    for( int i = 0; i < Size; i++ )
    {
        sum += (float)dA[i] * dA[i + shift];
    }
    dSum[shift] = sum;
}

int main( int argc, char* argv[ ] ){
    FILE *fp = fopen( "signal.txt", "r" );
    if( fp == NULL ){
        fprintf( stderr, "Cannot open file 'signal.txt'\n" );
        exit( 1 );
    }
    int Size;
    fscanf( fp, "%d", &Size );
    float *hA =     new float[ 2*Size ];
    float *hSum  = new float[ 1*Size ];

    for( int i = 0; i < Size; i++ ){
        fscanf( fp, "%f", &hA[i] );
        hA[i+Size] = hA[i];		// duplicate the array
    }
    fclose( fp );

	int dev = findCudaDevice(argc, (const char **)argv);

	// allocate host memory:
	float *dA, *dSum;

	dim3 dimArray(2*Size, 1, 1);
	dim3 dimSum(1*Size, 1, 1);

	cudaError_t status;
	status = cudaMalloc((void **)(&dA), 2*Size*sizeof(float));
	checkCudaErrors(status);

	status = cudaMalloc((void **)(&dSum), 1*Size*sizeof(float));
	checkCudaErrors(status);

	// copy host memory to the device:
	status = cudaMemcpy( dA, hA, 2*Size*sizeof(float), cudaMemcpyHostToDevice );
	checkCudaErrors( status );

    //Nothing to copy ?
	// status = cudaMemcpy( dSum, hSum, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
	// checkCudaErrors( status );

	// setup the execution parameters:
	dim3 threads(BLOCKSIZE, 1, 1 );
	dim3 grid(Size/threads.x, 1, 1 );

	// create and start timer
	cudaDeviceSynchronize( );

	// allocate CUDA events that we'll use for timing:
	cudaEvent_t start, stop;
	status = cudaEventCreate( &start );
	checkCudaErrors( status );
	status = cudaEventCreate( &stop );
	checkCudaErrors( status );

	// record the start event:
	status = cudaEventRecord( start, NULL );
	checkCudaErrors( status );

	// execute the kernel:
	AutoCorrelate<<< grid, threads >>>(dA, dSum);

	// record the stop event:
	status = cudaEventRecord( stop, NULL );
	checkCudaErrors( status );

	// wait for the stop event to complete:

	status = cudaEventSynchronize( stop );
	checkCudaErrors( status );

	float msecTotal = 0.0f;
	status = cudaEventElapsedTime( &msecTotal, start, stop );
	checkCudaErrors( status );

	// compute and print the performance
	double secondsTotal = 0.001 * (double)msecTotal;
	double megaCorsPerSecond = square(Size) / secondsTotal / 1000000.;
	fprintf( stderr, "Block size = %d, MegaTrials/Second = %10.4lf\n", BLOCKSIZE, megaCorsPerSecond );

	// copy result from the device to the host:
	status = cudaMemcpy( hSum, dSum, Size *sizeof(float), cudaMemcpyDeviceToHost );
	checkCudaErrors( status );

    if (BLOCKSIZE == 32) {
        FILE *fw = fopen( "CUDA.txt", "a" );
        for( int i = 1; i < 513; i++ ){
            fprintf( fw, "%d %f \n", i, hSum[i]);
        }
        fclose( fw );
    }
	// clean up memory:
	delete [ ] hA;
	delete [ ] hSum;

    status = cudaFree( dA );
    checkCudaErrors(status);
	status = cudaFree( dSum );
    checkCudaErrors(status);
	return 0;
}

double square (int x){
    return x*x;
}
