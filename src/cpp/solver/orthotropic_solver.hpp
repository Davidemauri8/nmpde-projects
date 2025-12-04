#pragma once
#ifndef __SOLVER_SUPERELASTIC_ISOTROPIC
#define __SOLVER_SUPERELASTIC_ISOTROPIC

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <functional>

using namespace dealii;

class SuperElasticOrthotropicSolver {

public:
	static constexpr unsigned int dim = 3;

	// Fourth order tensor acting on dimxdim matrices
	typedef Tensor<2, dim, double> SODTensor;

	SuperElasticOrthotropicSolver() = delete;

	SuperElasticOrthotropicSolver(
		const int _r,
		const double _ch_p,
		const double _alfa,
		const double _mu,
		const double _bulk,
		const double _af,
		const double _as,
		const double _asf
		// const std::function<FthODTensor(SODTensor&)> _depdef
	) :
		r(_r),
		ch_p(_ch_p),
		alfa(_alfa),
		mu(_mu),
		bulk(_bulk),
		af(_af),
		as(_as),
		asf(_asf)
	{ }

	void setup(const std::string& mesh);

	void solve();

	void output() const;

protected:

	typedef struct {

		Tensor<2, dim> ss0t;
		Tensor<2, dim> ff0t;
		double i4f;
		double i4s;

	} pass_cache_data_t;

	void
	voigt_apply_to(
		const Tensor<2, dim>&, const Tensor<2, dim>&,
		Tensor<2, dim>& into, const pass_cache_data_t&
	);

	void 
	orthothropic_base_at(
		const Point<dim>& p, std::vector<Tensor<1, dim>>& basis,
		bool compute_n
	);

	void
	compute_basis_at_quadrature(
		const std::vector<Point<dim>>& p,
		std::vector<std::vector<Tensor<1, dim>>>& orth_sys,
		bool compute_n
	);

	void compute_rh_s_newt_raphs();

	void build_system();

	// Triangulation.
	Triangulation<dim> mesh;

	// Finite element space.
	std::unique_ptr<FiniteElement<dim>> fe;

	// Quadrature formula.
	std::unique_ptr<Quadrature<dim>> quadrature;

	std::unique_ptr<Quadrature<dim - 1>> surf_quadrature;

	// DoF handler.
	DoFHandler<dim> dof_handler;

	// Sparsity pattern.
	SparsityPattern sparsity_pattern;

	// System solution.
	Vector<double> solution;

	Vector<double> step;

	Vector<double> nr_rhs_f;

	SparseMatrix<double> jacobian;

	AffineConstraints<double> constraints;

	// Polynomial degree.
	const unsigned int r;

	const double ch_p;

	const double alfa;

	const double mu;

	const double bulk;

	const double af;

	const double as;

	const double asf;

};


#endif // !__SOLVER_SUPERELASTIC_ISOTROPIC
