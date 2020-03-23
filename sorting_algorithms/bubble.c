#include <stdio.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>
#include <x86intrin.h>
#include "sorting.h"
#include <stdbool.h>

/* 
   bubble sort -- sequential, parallel -- 
*/
void sequential_bubble_sort (uint64_t *T, const uint64_t size)
{
    /* TODO: sequential implementation of bubble sort */ 
    bool tmpSorted = true;
    do {
        tmpSorted = true;
        for(int i = 0; i < size; i++) {
            if(T[i]>T[i+1]) {
                int tmpSwap = T[i];
                T[i] = T[i+1];
                T[i+1] = tmpSwap;
                tmpSorted = false;
            }
        }
    } while(tmpSorted == false);
    return ;
}

void parallel_bubble_sort (uint64_t *T, const uint64_t size, int nChunks, int nThreads)
{
    /* TODO: parallel implementation of bubble sort */
    bool tmpSorted = true;
    int chunkSize = size / nChunks;
    do {
        tmpSorted = true;
        #pragma omp parallel for schedule(dynamic) num_threads (nThreads)
        for (size_t k = 0; k < nChunks; k++) {
            for(int i = (chunkSize * k); i < ((chunkSize * (k+1))-1); i++){
                if(T[i]>T[i+1]) {
                    int tmpSwap = T[i];
                    T[i] = T[i+1];
                    T[i+1] = tmpSwap;
                    tmpSorted = false;
                }
            }
        }
        #pragma omp parallel for schedule(dynamic) num_threads (nThreads)
        for (int i = (chunkSize-1); i < size; i = i + chunkSize) {
            if (T[i]>T[i+1]) {
                int tmpSwap = T[i];
                T[i] = T[i+1];
                T[i+1] = tmpSwap;
                tmpSorted = false;
            }
        }
    } while(tmpSorted == false);
    return;
}

int main (int argc, char **argv)
{
    uint64_t start, end;
    uint64_t av ;
    unsigned int exp ;
    double Tsequential;
    double Tparallel;
    double speedups;

    /* the program takes one parameter N which is the size of the array to
       be sorted. The array will have size 2^N -> now it takes 3 */
    if (argc != 4) {
        fprintf (stderr, "bubble.run nSize nChunks nThreads \n") ;
        exit (-1) ;
    }

    uint64_t N = 1 << (atoi(argv[1])) ;
    int nChunks = atoi(argv[2]);
    int nThreads = atoi(argv[3]);
    uint64_t *X = (uint64_t *) malloc (N * sizeof(uint64_t)) ;
    printf("Bubble Sort :\n");
    printf("--> Sorting an array of size %lu\n",N);
    printf("--> Number of chunks %d\n",nChunks);
    printf("--> Number of threads %d\n",nThreads);
    printf("--> Number of experiments %d\n\n",NBEXPERIMENTS);
#ifdef RINIT
    printf("--> The array is initialized randomly\n");
#endif
    for (exp = 0 ; exp < NBEXPERIMENTS; exp++) {
#ifdef RINIT
        init_array_random (X, N);
#else
        init_array_sequence (X, N);
#endif
        // printf("Before : \n");
        // for (int i = 0; i < N; i++)
        // {
        //     printf("%ld ", X[i]);
        // }
        // printf("\n");
        start = _rdtsc () ;
        sequential_bubble_sort (X, N) ;
        end = _rdtsc () ;
        experiments [exp] = end - start ;

        /* verifying that X is properly sorted */
#ifdef RINIT
        if (! is_sorted (X, N)) {
            fprintf(stderr, "ERROR: the sequential sorting of the array failed\n") ;
            print_array (X, N) ;
            exit (-1) ;
	    }
#else
        if (! is_sorted_sequence (X, N)) {
            fprintf(stderr, "ERROR: the sequential sorting of the array failed\n") ;
            print_array (X, N) ;
            exit (-1) ;
	    }
#endif
    }

    av = average_time() ;
    Tsequential = (double)av/1000000;
    printf ("bubble serial \t\t\t %.2lf Mcycles\n", (double)av/1000000) ;
  
    for (exp = 0 ; exp < NBEXPERIMENTS; exp++) {
#ifdef RINIT
        init_array_random (X, N);
#else
        init_array_sequence (X, N);
#endif
        start = _rdtsc () ;
        parallel_bubble_sort (X, N, nChunks, nThreads) ;
        end = _rdtsc () ;
        experiments [exp] = end - start ;

        /* verifying that X is properly sorted */
#ifdef RINIT
        if (! is_sorted (X, N)) {
            fprintf(stderr, "ERROR: the parallel sorting of the array failed\n") ;
            exit (-1) ;
	    }
#else
        if (! is_sorted_sequence (X, N)) {
            fprintf(stderr, "ERROR: the parallel sorting of the array failed\n") ;
            exit (-1) ;
	    }
#endif          
    }

    av = average_time() ; 
    Tparallel =  (double)av/1000000;
    printf ("bubble parallel \t\t %.2lf Mcycles\n", (double)av/1000000) ;
    
    /* before terminating, we run one extra test of the algorithm */
    uint64_t *Y = (uint64_t *) malloc (N * sizeof(uint64_t)) ;
    uint64_t *Z = (uint64_t *) malloc (N * sizeof(uint64_t)) ;

#ifdef RINIT
    init_array_random (Y, N);
#else
    init_array_sequence (Y, N);
#endif
    memcpy(Z, Y, N * sizeof(uint64_t));
    sequential_bubble_sort (Y, N) ;
    parallel_bubble_sort (Z, N, nChunks, nThreads) ;

    if (! are_vector_equals (Y, Z, N)) {
        fprintf(stderr, "ERROR: sorting with the sequential and the parallel algorithm does not give the same result\n") ;
        exit (-1) ;
    }

    speedups = Tsequential/Tparallel;
    printf("Speedups \t\t\t %0.3f\n", speedups);

    free(X);
    free(Y);
    free(Z);
}
