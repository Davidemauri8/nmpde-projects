#pragma once
#ifndef _MATH_UTILS
#define _MATH_UTILS

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

FORCE_INLINE
double
pow_m2t(double v) {
    const double av = (v > 0) ? v : -v;
    const double k = (5 + av) / (1 + 5 * av);
    return (v > 0) ? k : -k;
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

#endif // !_MATH_UTILS
