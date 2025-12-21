#include "immintrin.h"

FORCE_INLINE
void 
do_voigt_product(
    const double* a,
    const double* x, double* y) {

    double final_sum;

    // --- Process each of the 9 rows of the matrix A ---
    for (int i = 0; i < 9; ++i)
    {
        __m256d sum_vec = _mm256_setzero_pd();
        __m256d a_vec1 = _mm256_loadu_pd(a); // Load A_i0 to A_i3
        __m256d x_vec1 = _mm256_loadu_pd(x);         // Load x_0 to x_3

        sum_vec = _mm256_fmadd_pd(a_vec1, x_vec1, sum_vec);

        // Load the next 4 elements (A_i4 to A_i7, x_4 to x_7)
        __m256d a_vec2 = _mm256_loadu_pd(a + 4);
        __m256d x_vec2 = _mm256_loadu_pd(x + 4);

        // sum_vec += a_vec2 * x_vec2
        sum_vec = _mm256_fmadd_pd(a_vec2, x_vec2, sum_vec);

        __m256d tmp1 = _mm256_hadd_pd(sum_vec, sum_vec);

        __m128d tmp2 = _mm256_extractf128_pd(tmp1, 0); // Extract low 128 bits (s0+s1)
        __m128d tmp3 = _mm256_extractf128_pd(tmp1, 1); // Extract high 128 bits (s2+s3)

        __m128d final_v = _mm_add_pd(tmp2, tmp3); // final_v = [s0+s1+s2+s3, s0+s1+s2+s3]

        _mm_store_sd(&final_sum, final_v); // Stores the dot product of A_i0..7 and x_0..7

        final_sum += a[8] * x[8];
        y[i] = final_sum;

        a += 9;
    }
    return;
}

