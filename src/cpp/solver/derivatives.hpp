#pragma once
#ifndef _DERIVATIVES_HPP
#define _DERIVATIVES_HPP

#include <deal.II/base/tensor.h>

template <unsigned int dim>
void
compute_m_of(
	const Tensor<2, dim>& F, Tensor<2, dim>& into
) {
	const auto inv = invert(F);
	for (int k = 0; k < 3; ++k)
		for (int l = 0; l < 3; ++l)
			into[k][l] = inv[k][i]
}

template <unsigned int dim>
void de_pbulk_deF(
	const Tensor<2, dim>& F
	Tensor<2, dim>& into
) {
	constexpr const double bulk = 1.23;

	const double J = determinant(F);
	const double Fmt = transpose(invert(F));

}





#endif