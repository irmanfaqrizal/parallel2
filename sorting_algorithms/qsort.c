#include <stdio.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>
#include <x86intrin.h>
#include "sorting.h"

static int compare (const void *x, const void *y)
{
    /* TODO: comparison function to be used by qsort()*/
    const int* castX = x;
    const int* castY = y;
    return (*castX > *castY) - (*castX < *castY);
}

void sequential_qsort_sort (uint64_t *T, const int size)
{
    /* TODO: sequential sorting based on libc qsort() function */
    qsort(T, size, sizeof(uint64_t), compare);
    return ;
}

/* 
   Merge two sorted chunks of array T!
   The two chunks are of size size
   First chunck starts at T[0], second chunck starts at T[size]
*/
void merge (uint64_t *T, const uint64_t size)
{
  uint64_t *X = (uint64_t *) malloc (2 * size * sizeof(uint64_t)) ;
  uint64_t i = 0 ;
  uint64_t j = size ;
  uint64_t k = 0 ;
  
  while ((i < size) && (j < 2*size)) {
    if (T[i] < T [j]) {
      X [k] = T [i] ;
      i = i + 1 ;
    } else {
      X [k] = T [j] ;
      j = j + 1 ;
    }
    k = k + 1 ;
  }

  if (i < size) {
    for (; i < size; i++, k++) {
      X [k] = T [i] ;
    }
  } else {
    for (; j < 2*size; j++, k++) {
      X [k] = T [j] ;
    }
  }
  memcpy (T, X, 2*size*sizeof(uint64_t)) ;
  free (X) ;
  return ;
}

void parallel_qsort_sort (uint64_t *T, const uint64_t size, int nChunks, int nThreads)
{
    /* TODO: parallel sorting based on libc qsort() function +
     * sequential merging */
    int chunkSize = size / nChunks;
    #pragma omp parallel for schedule(dynamic) num_threads (nThreads)
    for (size_t k = 0; k < nChunks; k++) {
      qsort(T + (k*chunkSize), chunkSize, sizeof(uint64_t), compare);
    }
    int chunksLeft = nChunks;
    int chunksSizeAtK = chunkSize;
    while (chunksLeft > 1) {
      for (size_t k = 0; k < chunksLeft; k=k+2) {
        merge (T + (k*chunksSizeAtK), chunksSizeAtK);
      }
      chunksLeft = chunksLeft / 2;
      chunksSizeAtK = chunksSizeAtK * 2;
    }
}

void parallel_qsort_sort1 (uint64_t *T, const uint64_t size, int nChunks, int nThreads)
{
    /* TODO: parallel sorting based on libc qsort() function +
     * PARALLEL merging */
    int chunkSize = size / nChunks;
    #pragma omp parallel for schedule(dynamic) num_threads (nThreads)
    for (size_t k = 0; k < nChunks; k++) {
      qsort(T + (k*chunkSize), chunkSize, sizeof(uint64_t), compare);
    }
    int chunksLeft = nChunks;
    int chunksSizeAtK = chunkSize;
    while (chunksLeft > 1) {
      #pragma omp parallel for schedule(dynamic) num_threads (nThreads)
      for (size_t k = 0; k < chunksLeft; k=k+2) {
        merge (T + (k*chunksSizeAtK), chunksSizeAtK);
      }
      chunksLeft = chunksLeft / 2;
      chunksSizeAtK = chunksSizeAtK * 2;
    }
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
       be sorted. The array will have size 2^N */
    if (argc != 4) {
        fprintf (stderr, "qsort.run nSize nChunks nThreads \n") ;
        exit (-1) ;
    }

    uint64_t N = 1 << (atoi(argv[1])) ;
    int nChunks = atoi(argv[2]);
    int nThreads = atoi(argv[3]);
    /* the array to be sorted */
    uint64_t *X = (uint64_t *) malloc (N * sizeof(uint64_t)) ;
    printf("Quick Sort :\n");
    printf("--> Sorting an array of size %lu\n",N);
    printf("--> Number of chunks %d\n",nChunks);
    printf("--> Number of threads %d\n",nThreads);
    printf("--> Number of experiments %d\n\n",NBEXPERIMENTS);
#ifdef RINIT
    printf("--> The array is initialized randomly\n");
#endif
    for (exp = 0 ; exp < NBEXPERIMENTS; exp++){
#ifdef RINIT
        init_array_random (X, N);
#else
        init_array_sequence (X, N);
#endif    
        start = _rdtsc () ;
        sequential_qsort_sort (X, N) ;
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
    printf ("qsort serial \t\t\t %.2lf Mcycles\n", (double)av/1000000) ;
  
    for (exp = 0 ; exp < NBEXPERIMENTS; exp++) {
#ifdef RINIT
      init_array_random (X, N);
#else
      init_array_sequence (X, N);
#endif  
      start = _rdtsc () ;  
      parallel_qsort_sort (X, N, nChunks, nThreads) ;     
      end = _rdtsc () ;
      experiments [exp] = end - start ;

      /* verifying that X is properly sorted */
#ifdef RINIT
      if (! is_sorted (X, N))
      {
          fprintf(stderr, "ERROR: the parallel sorting of the array failed\n") ;
          exit (-1) ;
	    }
#else
      if (! is_sorted_sequence (X, N))
      {
          fprintf(stderr, "ERROR: the parallel sorting of the array failed\n") ;
          exit (-1) ;
	    }
#endif            
    }
    
    av = average_time() ;  
    printf ("qsort parallel (merge seq) \t %.2lf Mcycles\n", (double)av/1000000) ;

    for (exp = 0 ; exp < NBEXPERIMENTS; exp++) {
#ifdef RINIT
      init_array_random (X, N);
#else
      init_array_sequence (X, N);
#endif
      start = _rdtsc () ;
      parallel_qsort_sort1 (X, N, nChunks, nThreads) ;
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
    Tparallel = (double)av/1000000;
    printf ("qsort parallel \t\t\t %.2lf Mcycles\n", (double)av/1000000) ;

    /* before terminating, we run one extra test of the algorithm */
    uint64_t *Y = (uint64_t *) malloc (N * sizeof(uint64_t)) ;
    uint64_t *Z = (uint64_t *) malloc (N * sizeof(uint64_t)) ;
#ifdef RINIT
    init_array_random (Y, N);
#else
    init_array_sequence (Y, N);
#endif
    memcpy(Z, Y, N * sizeof(uint64_t));
    sequential_qsort_sort (Y, N) ;
    parallel_qsort_sort1 (Z, N, nChunks, nThreads) ;

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
