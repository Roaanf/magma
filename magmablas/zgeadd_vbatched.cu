/*
    -- MAGMA (version 2.10.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver

       @precisions normal z -> s d c

       Custom extension: variable-size batched GEADD, not part of upstream MAGMA.
       Modeled after magmablas/zlacpy_vbatched.cu for the device kernel, and
       after magmablas/zgeadd_batched.cu for keeping everything -- device
       kernel, unchecked launcher, and the checked extern "C" host entry
       point -- in a single .cu file rather than splitting host/device code
       across files. nvcc compiles plain host C++ fine, so there's no need
       to split; a single file also avoids a forward-declaration that has
       to be kept in sync between two files for no benefit.

       IMPORTANT: m, n, ldda, lddb are DEVICE arrays (dimension batchCount),
       same convention as every other vbatched routine (gemm/gemv/syrk/...).
       The kernel indexes them itself via blockIdx.z. The checked entry point
       at the bottom of this file (magmablas_zgeadd_vbatched) does a small
       D2H copy of m[]/n[] only -- just enough to compute max_m/max_n for
       sizing the launch grid -- then calls the device-array launcher above.
*/
#include "magma_internal.h"

#define BLK_X 64
#define BLK_Y 32

/******************************************************************************/
/*
    Device worker for a single (m, n) matrix pair:
        dB = alpha*dA + dB

    Each thread owns one row (BLK_X threads per block along x) and walks
    BLK_Y columns, same striping used by zlacpy/zlaset device kernels.
*/
static __device__ void
zgeadd_device(
    int m, int n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex* dA, int ldda,
    magmaDoubleComplex*       dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;

    /* full block-column: skip the tail bounds check */
    bool full = (iby + BLK_Y <= n);

    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;

        if ( full ) {
            #pragma unroll
            for (int j = 0; j < BLK_Y; ++j) {
                dB[j*lddb] = alpha*dA[j*ldda] + dB[j*lddb];
            }
        }
        else {
            for (int j = 0; j < BLK_Y && iby+j < n; ++j) {
                dB[j*lddb] = alpha*dA[j*ldda] + dB[j*lddb];
            }
        }
    }
}


/******************************************************************************/
/*
    vbatched kernel: blockIdx.z indexes the matrix within the batch.
    m[], n[], ldda[], lddb[] are per-matrix; grid.x/grid.y are sized off
    max_m/max_n across the whole batch, so smaller matrices simply skip
    the blocks that fall outside their own (m[i], n[i]).
*/
__global__ void
zgeadd_vbatched_kernel(
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dAarray, magma_int_t* ldda,
    magmaDoubleComplex** dBarray, magma_int_t* lddb )
{
    const int batchid = blockIdx.z;
    const int my_m = (int)m[batchid];
    const int my_n = (int)n[batchid];

    /* this matrix is smaller than max_m/max_n -- drop out-of-range blocks */
    if ( blockIdx.x >= magma_ceildiv(my_m, BLK_X) ) return;
    if ( blockIdx.y >= magma_ceildiv(my_n, BLK_Y) ) return;

    zgeadd_device(
        my_m, my_n, alpha,
        dAarray[batchid], (int)ldda[batchid],
        dBarray[batchid], (int)lddb[batchid] );
}


/******************************************************************************/
/*
    Purpose
    -------
    ZGEADD_VBATCHED_MAX_NOCHECK adds two sets of general matrices with
    different dimensions, in a variable-size batched fashion:

        dB[i] = alpha*dA[i] + dB[i],   i = 0, ..., batchCount-1

    This is the unchecked launcher -- callers are expected to have already
    validated arguments and computed max_m, max_n (typically by copying m,n
    to the host once and scanning them, as magmablas_zgeadd_vbatched does).
    It is not meant to be called directly by users.

    Arguments
    ---------
    @param[in]
    m, n        INTEGER arrays, dimension(batchCount), on the GPU.
                Row/column counts of each matrix in the batch. Indexed
                directly by the kernel via blockIdx.z -- never dereferenced
                on the host, only pointer-offset (m+i) for batch chunking.

    @param[in]
    alpha       COMPLEX_16 scalar applied to each dA[i].

    @param[in]
    dAarray     Array of pointers, dimension(batchCount), on the GPU.
                Each dAarray[i] is a DOUBLE COMPLEX array on the GPU.

    @param[in]
    ldda        INTEGER array, dimension(batchCount), on the GPU.

    @param[in,out]
    dBarray     Array of pointers, dimension(batchCount), on the GPU.

    @param[in]
    lddb        INTEGER array, dimension(batchCount), on the GPU.

    @param[in]
    max_m, max_n  INTEGER
                  Maximum m and n across the batch; used to size the grid.

    @param[in]
    batchCount  INTEGER. Number of matrices to add.

    @param[in]
    queue       magma_queue_t. Queue to execute in.

    @ingroup magma_geadd_batched
*******************************************************************************/
extern "C" void
magmablas_zgeadd_vbatched_max_nocheck(
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dAarray, magma_int_t* ldda,
    magmaDoubleComplex              **dBarray, magma_int_t* lddb,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue )
{
    if ( max_m <= 0 || max_n <= 0 || batchCount <= 0 ) {
        return;
    }

    dim3 threads( BLK_X, 1, 1 );

    /* grid.z (batchCount) is capped by the CUDA/HIP launch limit (65535 on
       most archs); chunk the batch the same way gemm_vbatched does. */
    magma_int_t max_batchCount = queue->get_maxBatch();

    for (magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount - i);

        dim3 grid( magma_ceildiv(max_m, BLK_X),
                   magma_ceildiv(max_n, BLK_Y),
                   ibatch );

        zgeadd_vbatched_kernel
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m+i, n+i, alpha, dAarray+i, ldda+i, dBarray+i, lddb+i );
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZGEADD_VBATCHED adds two sets of general matrices, where the matrices
    in the batch may have different (m, n):

        dBarray[i] = alpha*dAarray[i] + dBarray[i],   i = 0, ..., batchCount-1

    This is the variable-size batched analogue of magmablas_zgeadd_batched.
    Unlike that fixed-size routine, m, n, ldda, and lddb are per-matrix and
    live on the GPU, exactly like magmablas_zgemm_vbatched.

    This is the checked entry point: it derives max_m/max_n from m[]/n[]
    (via a small D2H copy) and validates arguments before handing off to
    magmablas_zgeadd_vbatched_max_nocheck above.

    Arguments
    ---------
    @param[in]
    m       INTEGER array, dimension(batchCount), on the GPU.
            m[i] is the number of rows of the i-th matrix. m[i] >= 0.

    @param[in]
    n       INTEGER array, dimension(batchCount), on the GPU.
            n[i] is the number of columns of the i-th matrix. n[i] >= 0.

    @param[in]
    alpha   COMPLEX_16
            The scalar alpha applied to each dAarray[i].

    @param[in]
    dAarray Array of pointers, dimension(batchCount), on the GPU.
            The i-th pointer is a DOUBLE COMPLEX array, dimension(ldda[i], n[i]).

    @param[in]
    ldda    INTEGER array, dimension(batchCount), on the GPU.
            ldda[i] >= max(1, m[i]).

    @param[in,out]
    dBarray Array of pointers, dimension(batchCount), on the GPU.
            The i-th pointer is a DOUBLE COMPLEX array, dimension(lddb[i], n[i]).
            On exit, overwritten by alpha*dAarray[i] + dBarray[i].

    @param[in]
    lddb    INTEGER array, dimension(batchCount), on the GPU.
            lddb[i] >= max(1, m[i]).

    @param[in]
    batchCount  INTEGER
                The number of matrices to add. batchCount >= 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_geadd_batched
*******************************************************************************/
extern "C" void
magmablas_zgeadd_vbatched(
    magma_int_t* m, magma_int_t* n,        // device arrays
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dAarray, magma_int_t* ldda,  // device array
    magmaDoubleComplex              **dBarray, magma_int_t* lddb,   // device array
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( batchCount < 0 ) {
        info = -7;
    }
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    if ( batchCount == 0 ) {
        return;
    }

    // m, n live on the GPU -- pull them to the host just far enough to
    // compute max_m/max_n for the launch grid and do a basic sanity check.
    // (ldda/lddb are left on the device untouched; checking them properly
    // against m[] would need another D2H round trip and isn't done here --
    // see the README note on this.)
    magma_int_t* h_m = NULL;
    magma_int_t* h_n = NULL;
    magma_imalloc_cpu( &h_m, batchCount );
    magma_imalloc_cpu( &h_n, batchCount );

    magma_igetvector( batchCount, m, 1, h_m, 1, queue );
    magma_igetvector( batchCount, n, 1, h_n, 1, queue );

    magma_int_t max_m = 0, max_n = 0;
    for (magma_int_t i = 0; i < batchCount; ++i) {
        if ( h_m[i] < 0 ) {
            info = -1;
            break;
        }
        else if ( h_n[i] < 0 ) {
            info = -2;
            break;
        }
        max_m = max( max_m, h_m[i] );
        max_n = max( max_n, h_n[i] );
    }

    magma_free_cpu( h_m );
    magma_free_cpu( h_n );

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    if ( max_m == 0 || max_n == 0 ) {
        return;
    }

    magmablas_zgeadd_vbatched_max_nocheck(
        m, n, alpha,
        dAarray, ldda,
        dBarray, lddb,
        max_m, max_n, batchCount, queue );
}

