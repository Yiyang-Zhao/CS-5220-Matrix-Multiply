#include<string.h>
#include<stdlib.h>
#include <xmmintrin.h>
#include <immintrin.h> 
const char* dgemm_desc = "My awesome dgemm.";


#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/

void copy_block(const int lda, const double * matrix, double * matrix_block, const int M, const int N) {
    for (int i = 0; i < N; ++i) {
        memcpy(matrix_block + i * BLOCK_SIZE, matrix + i * lda, M * sizeof(double));
    }
}

void copy_block_C(const int lda, double * matrix, double * matrix_block, const int M, const int N) {
    for (int i = 0; i < N; ++i) {
        memcpy(matrix_block + i * BLOCK_SIZE, matrix + i * lda, M * sizeof(double));
    }
}

void copy_back(const int lda, double * matrix, double * matrix_block, const int M, const int N) {
    for (int i = 0; i < N; ++i) {
        memcpy(matrix + i * lda, matrix_block + i * BLOCK_SIZE, M * sizeof(double));
    }
}

// Copy block of B into a contiguous buffer
void copy_block_transposed(const int lda, const double * B, double * B_block, const int K, const int N) {
    for (int i = 0; i < N; ++i) {
        memcpy(B_block + i * K, B + i * lda, K * sizeof(double));
    }
}

// Optimized basic_dgemm using copied buffers
void basic_dgemm_optimized(const int M, const int N, const int K,
                           const double * restrict A_block, const double * restrict B_block, double * restrict C_block,
                           const int lda)
{
    const double *A_block_aligned = (const double *) __builtin_assume_aligned(A_block, 64);
    const double *B_block_aligned = (const double *) __builtin_assume_aligned(B_block, 64);
    double *C_block_aligned = (double *) __builtin_assume_aligned(C_block, 64);

    int i, j, k;

    // for (j = 0; j < N; ++j) {
    //     for (k = 0; k < K; ++k) {
    //         __m512d bkj = _mm512_set1_pd(B_block_aligned[j * BLOCK_SIZE + k]);  // Broadcast B[j, k]
    //         for (i = 0; i < M; i += 8) {  // Process 8 elements at a time
    //             __m512d a = _mm512_load_pd(&A_block_aligned[k * BLOCK_SIZE + i]);  // Load 8 elements of A[k, i:i+7]
    //             __m512d c = _mm512_load_pd(&C_block_aligned[j * BLOCK_SIZE + i]);      // Load 8 elements of C[j, i:i+7]
    //             c = _mm512_fmadd_pd(a, bkj, c);  // C[j, i:i+7] += A[k, i:i+7] * B[j, k]
    //             _mm512_store_pd(&C_block_aligned[j * BLOCK_SIZE + i], c);  // Store result back to C[j, i:i+7]
    //         }
    //     }
    // }

    for (j = 0; j < N; ++j) {
        for (k = 0; k < K; k += 8) {
            __m512d bkj0 = _mm512_set1_pd(B_block_aligned[j * BLOCK_SIZE + k]);  // Broadcast B[j, k]
            __m512d bkj1 = _mm512_set1_pd(B_block_aligned[j * BLOCK_SIZE + k + 1]);  // Broadcast B[j, k]
            __m512d bkj2 = _mm512_set1_pd(B_block_aligned[j * BLOCK_SIZE + k + 2]);  // Broadcast B[j, k]
            __m512d bkj3 = _mm512_set1_pd(B_block_aligned[j * BLOCK_SIZE + k + 3]);  // Broadcast B[j, k]
            __m512d bkj4 = _mm512_set1_pd(B_block_aligned[j * BLOCK_SIZE + k + 4]);  // Broadcast B[j, k]
            __m512d bkj5 = _mm512_set1_pd(B_block_aligned[j * BLOCK_SIZE + k + 5]);  // Broadcast B[j, k]
            __m512d bkj6 = _mm512_set1_pd(B_block_aligned[j * BLOCK_SIZE + k + 6]);  // Broadcast B[j, k]
            __m512d bkj7 = _mm512_set1_pd(B_block_aligned[j * BLOCK_SIZE + k + 7]);  // Broadcast B[j, k]
            
            for (i = 0; i < M; i += 8) {  // Process 16 elements at a time
                __m512d a0 = _mm512_load_pd(&A_block_aligned[k * BLOCK_SIZE + i]);    // Load 8 elements of A[k, i:i+7]
                __m512d a1 = _mm512_load_pd(&A_block_aligned[(k+1) * BLOCK_SIZE + i]); // Load next 8 elements of A[k, i+8:i+15]
                __m512d a2 = _mm512_load_pd(&A_block_aligned[(k+2) * BLOCK_SIZE + i]);    // Load 8 elements of A[k, i:i+7]
                __m512d a3 = _mm512_load_pd(&A_block_aligned[(k+3) * BLOCK_SIZE + i]); // Load next 8 elements of A[k, i+8:i+15]
                __m512d a4 = _mm512_load_pd(&A_block_aligned[(k+4) * BLOCK_SIZE + i]);    // Load 8 elements of A[k, i:i+7]
                __m512d a5 = _mm512_load_pd(&A_block_aligned[(k+5) * BLOCK_SIZE + i]); // Load next 8 elements of A[k, i+8:i+15]
                __m512d a6 = _mm512_load_pd(&A_block_aligned[(k+6) * BLOCK_SIZE + i]);    // Load 8 elements of A[k, i:i+7]
                __m512d a7 = _mm512_load_pd(&A_block_aligned[(k+7) * BLOCK_SIZE + i]); // Load next 8 elements of A[k, i+8:i+15]
                
                __m512d c0 = _mm512_load_pd(&C_block_aligned[j * BLOCK_SIZE + i]);      // Load 8 elements of C[j, i:i+7]
                
                c0 = _mm512_fmadd_pd(a0, bkj0, c0);  // C[j, i:i+7] += A[k, i:i+7] * B[j, k]
                c0 = _mm512_fmadd_pd(a1, bkj1, c0);  // C[j, i:i+7] += A[k, i:i+7] * B[j, k]
                c0 = _mm512_fmadd_pd(a2, bkj2, c0);  // C[j, i:i+7] += A[k, i:i+7] * B[j, k]
                c0 = _mm512_fmadd_pd(a3, bkj3, c0);  // C[j, i:i+7] += A[k, i:i+7] * B[j, k]
                c0 = _mm512_fmadd_pd(a4, bkj4, c0);  // C[j, i:i+7] += A[k, i:i+7] * B[j, k]
                c0 = _mm512_fmadd_pd(a5, bkj5, c0);  // C[j, i:i+7] += A[k, i:i+7] * B[j, k]
                c0 = _mm512_fmadd_pd(a6, bkj6, c0);  // C[j, i:i+7] += A[k, i:i+7] * B[j, k]
                c0 = _mm512_fmadd_pd(a7, bkj7, c0);  // C[j, i:i+7] += A[k, i:i+7] * B[j, k]
                
                _mm512_store_pd(&C_block_aligned[j * BLOCK_SIZE + i], c0);  // Store result back to C[j, i:i+7]
                // _mm512_store_pd(&C_block_aligned[j * BLOCK_SIZE + i + 8], c1); // Store result back to C[j, i+8:i+15]
                // _mm512_store_pd(&C_block_aligned[j * BLOCK_SIZE + i + 16], c2);  // Store result back to C[j, i:i+7]
                // _mm512_store_pd(&C_block_aligned[j * BLOCK_SIZE + i + 24], c3); // Store result back to C[j, i+8:i+15]
            }
        }
    }


    
}   


// Perform matrix multiplication on a block of A, B, and C
void do_block(const int lda, const double * A, const double * B, double * C_block,
              const int i, const int j, const int k)
{
    const int M = (i + BLOCK_SIZE > lda ? lda - i : BLOCK_SIZE);
    const int N = (j + BLOCK_SIZE > lda ? lda - j : BLOCK_SIZE);
    const int K = (k + BLOCK_SIZE > lda ? lda - k : BLOCK_SIZE);

    // Allocate aligned memory for blocks of A and B
    double *A_block = (double *)_mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double), 64);
    if (A_block != NULL) 
        memset(A_block, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    // double *C_block = (double *)_mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double), 64);
    // if (C_block != NULL) 
    //     memset(C_block, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    double *B_block = (double *)_mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double), 64);
    if (B_block != NULL) 
        memset(B_block, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

    // Copy blocks of A and B into contiguous memory
    copy_block(lda, A + i + k * lda, A_block, M, K);
    copy_block(lda, B + k + j * lda, B_block, K, N);
    // copy_block_C(lda, C + i + j * lda, C_block, M, N);

    // Perform matrix multiplication on copied blocks
    basic_dgemm_optimized(M, N, K, A_block, B_block, C_block, lda);

    // copy_back(lda, C + i + j * lda, C_block, M, N);

    // Free the aligned memory
    _mm_free(A_block);
    _mm_free(B_block);
    // _mm_free(C_block);
}

// Main matrix multiplication function
void square_dgemm(const int M, const double * A, const double * B, double * C)
{
    const int n_blocks = M / BLOCK_SIZE + (M % BLOCK_SIZE ? 1 : 0);
    int bi, bj, bk;

    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            int CM = (i + BLOCK_SIZE > M ? M - i : BLOCK_SIZE);
            int CN = (j + BLOCK_SIZE > M ? M - j : BLOCK_SIZE);
            double *C_block = (double *)_mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double), 64);
            if (C_block != NULL) 
                memset(C_block, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
            // copy_block_C(M, C + i + j * M, C_block, CM, CN);
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C_block, i, j, k);
            }
            copy_back(M, C + i + j * M, C_block, CM, CN);
            _mm_free(C_block);
        }
    }
}