#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <omp.h>
#include <xmmintrin.h>

#ifndef SSE_WIDTH
#define SSE_WIDTH	4
#endif

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES	10
#endif

const int Size = 32769;

//Function Prototypes
double square (int);
float SimdMulSum(float *, float *, int);

int main (){
    FILE *fp = fopen( "signal.txt", "r" );
    if( fp == NULL ){
        fprintf( stderr, "Cannot open file 'signal.txt'\n" );
        exit( 1 );
    }
    int Size;
    fscanf( fp, "%d", &Size );
    float *A =     new float[ 2*Size ];
    float *Sums  = new float[ 1*Size ];

    for( int i = 0; i < Size; i++ ){
        fscanf( fp, "%f", &A[i] );
        A[i+Size] = A[i];		// duplicate the array
    }
    fclose( fp );

    float maxPerformance = 0.;      // must be declared outside the NUMTRIES loop
    // looking for the maximum performance:
    for( int t = 0; t < NUMTRIES; t++ ){
    double time0 = omp_get_wtime( );

    for( int shift = 0; shift < Size; shift++ ){

        Sums[shift] = SimdMulSum(&A[0], &A[0+shift], Size);
    }

    double time1 = omp_get_wtime( );
    double megaCorsPerSecond = square(Size) / ( time1 - time0 ) / 1000000.;
    if( megaCorsPerSecond > maxPerformance )
        maxPerformance = megaCorsPerSecond;
    }
    printf("%d\t, %8.15f\n", NUMTRIES, maxPerformance);

    if (SSE_WIDTH == 4 && NUMTRIES == 10){
        FILE *fw = fopen( "SIMD.txt", "a" );
        for( int i = 1; i < 513; i++ ){
            fprintf( fw, "%d %f \n", i, Sums[i]);
        }
        fclose( fw );
    }
    delete [] A;
    delete [] Sums;

    return 0;
}

double square (int x){
    return (double)(x*x);
}

float SimdMulSum(float *a, float *b, int len){
	float sum[4] = {0., 0., 0., 0.};
	int limit = (len/SSE_WIDTH) * SSE_WIDTH;
	register float *pa = a;
	register float *pb = b;

	__m128 ss = _mm_loadu_ps(&sum[0]);
	for(int i = 0; i < limit; i += SSE_WIDTH){
		ss = _mm_add_ps(ss, _mm_mul_ps(_mm_loadu_ps(pa), _mm_loadu_ps(pb)));
		pa += SSE_WIDTH;
		pb += SSE_WIDTH;
	}
	_mm_storeu_ps(&sum[0], ss);

	for(int i = limit; i < len; i++){
		sum[0] += a[i] * b[i];
	}

	return sum[0] + sum[1] + sum[2] + sum[3];
}