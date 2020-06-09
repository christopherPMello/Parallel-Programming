#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <omp.h>

// setting the number of threads:
#ifndef NUMT
#define NUMT		1
#endif

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES	10
#endif

const int Size = 32769;

//Function Prototypes
double square (int);

int main (){

    #ifndef _OPENMP
        fprintf( stderr, "OpenMP is not supported\n" );
        return 1;
    #endif

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

    omp_set_num_threads(NUMT);

    float maxPerformance = 0.;      // must be declared outside the NUMTRIES loop
    // looking for the maximum performance:
    for( int t = 0; t < NUMTRIES; t++ ){
        
    double time0 = omp_get_wtime( );

    #pragma omp parallel for default(none) shared(Size, A, Sums)
    for( int shift = 0; shift < Size; shift++ )
    {
        float sum = 0.;
        for( int i = 0; i < Size; i++ )
        {
            sum += A[i] * A[i + shift];
        }
        Sums[shift] = sum;	// note the "fix #2" from false sharing if you are using OpenMP
    }
    double time1 = omp_get_wtime( );
    double megaCorsPerSecond = square(Size) / ( time1 - time0 ) / 1000000.;
    if( megaCorsPerSecond > maxPerformance )
        maxPerformance = megaCorsPerSecond;
    }
    printf("%d\t, %d\t, %8.15f\n", NUMT, NUMTRIES, maxPerformance);

    if (NUMT == 1 && NUMTRIES == 10){
        FILE *fw = fopen( "SIMDT.txt", "a" );
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