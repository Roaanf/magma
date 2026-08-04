/*
    -- MAGMA (version 2.10.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver

       @precisions normal z -> c d s

       Custom extension: tester for magmablas_zgeadd_vbatched, not part of
       upstream MAGMA. Modeled on testing/testing_zgeadd_batched.cpp for the
       overall structure (timing, LAPACK-norm-based error check, opts.check
       branch) and on the random-per-matrix-size generation pattern used by
       testing/testing_zgetrf_vbatched.cpp for the variable-size part.

       NOTE: written against MAGMA's general testing conventions (magma_opts,
       TESTING_CHECK, magma_{set,get}vector, magma_{i,z}malloc[_cpu]). Field
       names on magma_opts (e.g. opts.align, opts.queue) can drift slightly
       between MAGMA releases -- check against your local testings.h if this
       doesn't compile cleanly out of the box.
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"


/******************************************************************************/
int main( int argc, char** argv )
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double          error, work[1];
    magmaDoubleComplex alpha = MAGMA_Z_MAKE( 1.5, -2.3 );

    magmaDoubleComplex *h_A=NULL, *h_B=NULL, *h_Bmagma=NULL;
    magmaDoubleComplex_ptr d_A=NULL, d_B=NULL;
    magmaDoubleComplex **h_A_array=NULL, **h_B_array=NULL;
    magmaDoubleComplex **d_A_array=NULL, **d_B_array=NULL;

    magma_int_t *h_M=NULL, *h_N=NULL, *h_ldda=NULL, *h_lddb=NULL;
    magma_int_t *h_Aoff=NULL, *h_Boff=NULL;   // per-matrix offsets into flat h_A/h_B
    magma_int_t *d_M=NULL, *d_N=NULL, *d_ldda=NULL, *d_lddb=NULL;

    magma_int_t M, N, batchCount;
    magma_int_t sizeA, sizeB, total_elems;
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("%% Variable-size batched GEADD: dB[i] = alpha*dA[i] + dB[i]\n");
    printf("%% M, N below are the MAX size per batch entry; actual m[i],n[i] are randomized in [1,M],[1,N]\n");
    printf("%%   M     N  batchCount    CPU GB/s (ms)      GPU GB/s (ms)     |Bmagma - Bref| / |Bref|\n");
    printf("%%======================================================================================\n");

    for (int itest = 0; itest < opts.ntest; ++itest) {
        for (int iter = 0; iter < opts.niter; ++iter) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            batchCount = opts.batchcount;

            TESTING_CHECK( magma_imalloc_cpu( &h_M,    batchCount ) );
            TESTING_CHECK( magma_imalloc_cpu( &h_N,    batchCount ) );
            TESTING_CHECK( magma_imalloc_cpu( &h_ldda, batchCount ) );
            TESTING_CHECK( magma_imalloc_cpu( &h_lddb, batchCount ) );
            TESTING_CHECK( magma_imalloc_cpu( &h_Aoff, batchCount ) );
            TESTING_CHECK( magma_imalloc_cpu( &h_Boff, batchCount ) );

            // --- randomize per-matrix (m, n), bounded by the -N M,N given on
            //     the command line for this itest; lay out a flat buffer with
            //     each matrix's own leading dimension.
            sizeA = 0;
            sizeB = 0;
            total_elems = 0;
            for (magma_int_t i = 0; i < batchCount; ++i) {
                h_M[i] = 1 + (magma_int_t)(rand() % M);
                h_N[i] = 1 + (magma_int_t)(rand() % N);
                h_ldda[i] = magma_roundup( h_M[i], opts.align );
                h_lddb[i] = magma_roundup( h_M[i], opts.align );

                h_Aoff[i] = sizeA;
                h_Boff[i] = sizeB;

                sizeA += h_ldda[i] * h_N[i];
                sizeB += h_lddb[i] * h_N[i];
                total_elems += h_M[i] * h_N[i];
            }

            TESTING_CHECK( magma_zmalloc_cpu( &h_A,      sizeA ) );
            TESTING_CHECK( magma_zmalloc_cpu( &h_B,      sizeB ) );
            TESTING_CHECK( magma_zmalloc_cpu( &h_Bmagma, sizeB ) );

            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );

            TESTING_CHECK( magma_zmalloc( &d_A, sizeA ) );
            TESTING_CHECK( magma_zmalloc( &d_B, sizeB ) );

            magma_zsetvector( sizeA, h_A, 1, d_A, 1, opts.queue );
            magma_zsetvector( sizeB, h_B, 1, d_B, 1, opts.queue );

            // --- build the array-of-pointers on the host, then push to device
            TESTING_CHECK( magma_malloc_cpu( (void**)&h_A_array, batchCount*sizeof(magmaDoubleComplex*) ) );
            TESTING_CHECK( magma_malloc_cpu( (void**)&h_B_array, batchCount*sizeof(magmaDoubleComplex*) ) );
            for (magma_int_t i = 0; i < batchCount; ++i) {
                h_A_array[i] = d_A + h_Aoff[i];
                h_B_array[i] = d_B + h_Boff[i];
            }
            TESTING_CHECK( magma_malloc( (void**)&d_A_array, batchCount*sizeof(magmaDoubleComplex*) ) );
            TESTING_CHECK( magma_malloc( (void**)&d_B_array, batchCount*sizeof(magmaDoubleComplex*) ) );
            magma_setvector( batchCount, sizeof(magmaDoubleComplex*), h_A_array, 1, d_A_array, 1, opts.queue );
            magma_setvector( batchCount, sizeof(magmaDoubleComplex*), h_B_array, 1, d_B_array, 1, opts.queue );

            // --- push per-matrix m, n, ldda, lddb to the device (this routine
            //     requires them there -- see zgeadd_vbatched.cpp)
            TESTING_CHECK( magma_imalloc( &d_M,    batchCount ) );
            TESTING_CHECK( magma_imalloc( &d_N,    batchCount ) );
            TESTING_CHECK( magma_imalloc( &d_ldda, batchCount ) );
            TESTING_CHECK( magma_imalloc( &d_lddb, batchCount ) );
            magma_setvector( batchCount, sizeof(magma_int_t), h_M,    1, d_M,    1, opts.queue );
            magma_setvector( batchCount, sizeof(magma_int_t), h_N,    1, d_N,    1, opts.queue );
            magma_setvector( batchCount, sizeof(magma_int_t), h_ldda, 1, d_ldda, 1, opts.queue );
            magma_setvector( batchCount, sizeof(magma_int_t), h_lddb, 1, d_lddb, 1, opts.queue );

            /* ====================================================================
               Run on the GPU
               =================================================================== */
            gpu_time = magma_sync_wtime( opts.queue );
            magmablas_zgeadd_vbatched(
                d_M, d_N, alpha,
                d_A_array, d_ldda,
                d_B_array, d_lddb,
                batchCount, opts.queue );
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;

            magma_zgetvector( sizeB, d_B, 1, h_Bmagma, 1, opts.queue );

            // memory-bound kernel (read A, read B, write B) -> report GB/s,
            // same convention MAGMA uses for lacpy/laset-style testers
            gbytes = (total_elems * 3.0 * sizeof(magmaDoubleComplex)) / 1e9;
            gpu_perf = gbytes / gpu_time;

            /* ====================================================================
               CPU reference
               =================================================================== */
            cpu_time = magma_wtime();
            for (magma_int_t i = 0; i < batchCount; ++i) {
                magmaDoubleComplex* A_i = h_A + h_Aoff[i];
                magmaDoubleComplex* B_i = h_B + h_Boff[i];
                magma_int_t ldda_i = h_ldda[i];
                magma_int_t lddb_i = h_lddb[i];
                for (magma_int_t col = 0; col < h_N[i]; ++col) {
                    for (magma_int_t row = 0; row < h_M[i]; ++row) {
                        magma_int_t ia = row + col*ldda_i;
                        magma_int_t ib = row + col*lddb_i;
                        // double2 has no operator*/operator+ in host C++ --
                        // use MAGMA's complex-arithmetic macros, same as the
                        // rest of MAGMA's CPU reference code in the testers.
                        B_i[ib] = MAGMA_Z_ADD( MAGMA_Z_MUL( alpha, A_i[ia] ), B_i[ib] );
                    }
                }
            }
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;

            /* ====================================================================
               Check
               =================================================================== */
            if ( opts.check ) {
                error = 0;
                for (magma_int_t i = 0; i < batchCount; ++i) {
                    magma_int_t m_i = h_M[i], n_i = h_N[i], ld_i = h_lddb[i];
                    magmaDoubleComplex* Bref = h_B      + h_Boff[i];
                    magmaDoubleComplex* Bmag = h_Bmagma + h_Boff[i];

                    double Bnorm = lapackf77_zlange( "F", &m_i, &n_i, Bref, &ld_i, work );

                    magma_int_t mn = ld_i * n_i;
                    magmaDoubleComplex neg_one = MAGMA_Z_NEG_ONE;
                    blasf77_zaxpy( &mn, &neg_one, Bref, &ione, Bmag, &ione );
                    double diffnorm = lapackf77_zlange( "F", &m_i, &n_i, Bmag, &ld_i, work );

                    double err_i = (Bnorm > 0) ? diffnorm / Bnorm : diffnorm;
                    error = max( error, err_i );
                }
                bool okay = (error < tol);
                status += ! okay;
                printf("%5lld %5lld %10lld     %7.2f (%7.2f)    %7.2f (%7.2f)    %8.2e   %s\n",
                       (long long) M, (long long) N, (long long) batchCount,
                       cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                       error, (okay ? "ok" : "failed") );
            }
            else {
                printf("%5lld %5lld %10lld       ---   (  ---  )    %7.2f (%7.2f)      ---\n",
                       (long long) M, (long long) N, (long long) batchCount,
                       gpu_perf, gpu_time*1000. );
            }

            magma_free_cpu( h_M );    magma_free_cpu( h_N );
            magma_free_cpu( h_ldda ); magma_free_cpu( h_lddb );
            magma_free_cpu( h_Aoff ); magma_free_cpu( h_Boff );
            magma_free_cpu( h_A );    magma_free_cpu( h_B ); magma_free_cpu( h_Bmagma );
            magma_free_cpu( h_A_array ); magma_free_cpu( h_B_array );

            magma_free( d_A );       magma_free( d_B );
            magma_free( d_A_array ); magma_free( d_B_array );
            magma_free( d_M );       magma_free( d_N );
            magma_free( d_ldda );    magma_free( d_lddb );

            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf("\n");
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
