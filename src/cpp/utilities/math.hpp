#pragma once
#ifndef _MATH_UTILS
#define _MATH_UTILS

#include <cmath>
#include <new> 
#include <deal.II/base/tensor.h>

#if defined(_MSC_VER)
// Microsoft Visual C++ Compiler
#define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
// GCC or Clang
#define FORCE_INLINE __attribute__((always_inline)) inline
#else
// Fallback for other compilers
#define FORCE_INLINE inline
#endif

using namespace dealii;

/*


*/

constexpr std::size_t cache_line_size() {
#ifdef SET_L1_CACHE_LINE_SIZE
    return SET_L1_CACHE_LINE_SIZE;
#else
#if __cplusplus > 201703L
    return std::hardware_destructive_interference_size;
#endif
    // Manually tuned to the programmer of this unit (run "getconf LEVEL1_DCACHE_LINESIZE"
    // on the terminal)
    return 64;
#endif
}

FORCE_INLINE
double
pow_m2t(double v) {
    // const double av = (v > 0) ? v : -v;
    // const double k = (5 + av) / (1 + 5 * av);
    return std::pow(v, -2 / 3.0);
    // return (v > 0) ? k : -k;
}


FORCE_INLINE
double
macaulay(double arg) {
    return (arg > 0.0) ? arg : 0.0;
}

template <unsigned int Dim>
FORCE_INLINE
Tensor<2, Dim>outer_product(const Tensor<1, Dim> v1, const Tensor<1, Dim> v2) {
    Tensor<2, Dim> t;
    t[0][0] = v1[0] * v2[0];
    t[0][1] = t[1][0] = v1[0] * v2[1];
    t[0][2] = t[2][0] = v1[0] * v2[2];
    t[1][1] = v1[1] * v2[1];
    t[1][2] = t[2][1] = v1[1] * v2[2];
    t[2][2] = v1[2] * v1[2];
    return t;
}


Tensor<2, 3>
tensor_product(const std::vector<Tensor<2, 3>>& fourth_rank, const Tensor<2, 3>& second_rank) {
    Tensor<2, 3> retval;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            retval[i][j] = scalar_product(fourth_rank[3 * i + j], second_rank);
        }
    return retval;
}

#endif // !_MATH_UTILS
