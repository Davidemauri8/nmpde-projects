#pragma once
#ifndef _DERIVATIVES_HPP
#define _DERIVATIVES_HPP

#define PDE_OUT_VERBOSE

#include <cmath>

#include <deal.II/base/tensor.h>

#include "../utilities/mesh_io.hpp"
#include "../utilities/table_io.hpp"

using namespace dealii;

namespace Validation {


	template <int dim>
	void
		compute_m_of(
			const Tensor<2, dim>& F, Tensor<2, dim>& into,
			const int i, const int j
		) {
		const auto inv = invert(F);
		for (int k = 0; k < 3; ++k)
			for (int l = 0; l < 3; ++l)
				into[k][l] = -inv[k][i] * inv[j][l];
	}

	constexpr const double k_bulk = 23.15;


	template <int dim>
	void de_pbulk_deF(
		const int i, const int j,
		const Tensor<2, dim>& F,
		Tensor<2, dim>& into
	) {
		Tensor<2, dim> m;

		const double J = determinant(F);
		const auto Fmt = transpose(invert(F));
		compute_m_of(F, m, i, j);

		into = k_bulk * J * J * Fmt * Fmt + (k_bulk / 2) * (J * J - 1) * transpose(m);

	}

	template <int dim>
	void pbulk(
		const Tensor<2, dim>& F,
		Tensor<2, dim>& into
	) {
		const double J = determinant(F);
		const auto Fmt = transpose(invert(F));

		into = (k_bulk / 2) * (J * J - 1) * Fmt;
	}

	constexpr const double a = 1;
	constexpr const double b = 1;

	template <int Dim>
	void piso(
		const Tensor<2, Dim>& F,
		Tensor<2, Dim>& into
	) {
		const double J = determinant(F);
		const auto Fmt = transpose(invert(F));
		const double Jm23 = pow(J, -2 / 3.0);
		const double I1 = Jm23 * trace(transpose(F)*F);

		into = a*exp(b*(I1-3)) * Jm23 * (F - (1 / 3.0) * Fmt);
	}
	
	template <int Dim>
	void
	compute_eiej(
		Tensor<2, Dim>& eiej, const int i, const int j
	) {
		eiej = 0.0;
		eiej[i][j] = 1.0;
		return;
	}


	template <int dim>
	void de_piso_deF(
		const int i, const int j,
		const Tensor<2, dim>& F,
		Tensor<2, dim>& into
	) {
		Tensor<2, dim> m, eiej;
		const double J = determinant(F);
		const auto Fmt = transpose(invert(F));
		const double Jm23 = pow(J, -2 / 3.0);
		const double I1 = Jm23 * trace(transpose(F) * F);
		compute_m_of(F, m, i, j);
		compute_eiej(eiej, i, j);

		// pde_out_c_var(Jm23, RED_COLOR);
		// pde_out_c_var(I1, RED_COLOR);
		// pde_out_c_var(J, RED_COLOR);
		// pde_out_c_var(eiej, RED_COLOR);
		// pde_out_c_var(Fmt, RED_COLOR);

		into = a * exp(b * (I1 - 3)) * Jm23 * (2 * b * Jm23 * (F - (1 / 3.0) * Fmt) * (F[i][j] -(1/3.0)*Fmt[i][j]*I1 -(2/3.0)*Fmt[i][j] ) + eiej - (1 / 3.0) * transpose(m));
		// pde_out_c_var(into, RED_COLOR);

	}

	template <int Dim>
	void
		random_init(Tensor<2, Dim>& t) {
		srand((unsigned)time(NULL));
		for (int i = 0; i < Dim; i++)
			for (int j = 0; j < Dim; ++j)
				t[i][j] = (double)rand() / ((double)RAND_MAX);
	}

	template <int Dim>
	double
		tensor_reduce_sum(
			const Tensor<2, Dim>& ref
		) {
		double d = 0.0;
		for (int i = 0; i < Dim; ++i)
			for (int j = 0; j < Dim; ++j) {
				d += ref[i][j];
			}
		return d;
	}

	template <int Dim>
	void
		tensor_elementwise_summation(
			Tensor<2, Dim>& ref, const double d
		) {
		for (int i = 0; i < Dim; ++i)
			for (int j = 0; j < Dim; ++j) {
				ref[i][j] += d;
			}
		return;
	}

	void
	verify_derivative()
	{
		double epsilon = 1e-3;
		Tensor<2, 3> test, symbolic_differential, computed_differential;
		Tensor<2, 3> Mbefore, Mafter, der;

		random_init(test);
		// Uncomment to test on norm 1 samples
		// test /= test.norm();

		pde_out("Test matrix" << test);

		for (int refinement = 0; refinement < 15; ++refinement) {

			for (int i = 0; i < 3; ++i)
				for (int j = 0; j < 3; ++j) {
					de_piso_deF(i, j, test, der);
					computed_differential[i][j] = epsilon * tensor_reduce_sum(der);
				}
			piso(test, Mbefore);
			// After this line test will be incremented by epsilon over all entries
			tensor_elementwise_summation(test, epsilon);
			piso(test, Mafter);

			symbolic_differential = Mafter - Mbefore;

			double diff = (symbolic_differential - computed_differential).norm();
			pde_out_c("Convergence: " << diff, RED_COLOR);
			diff = ((symbolic_differential - computed_differential)).norm() / epsilon;
			pde_out_c("Convergence: " << diff, RED_COLOR);

			epsilon /= 10.0;
		}
	}

}


#endif