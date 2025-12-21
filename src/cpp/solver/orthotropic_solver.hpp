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

using namespace dealii;

class SuperElasticOrthotropicSolver {

public:
	static constexpr unsigned int dim = 3;

	// Fourth order tensor acting on dimxdim matrices
	typedef Tensor<2, dim, double> SODTensor;

	SuperElasticOrthotropicSolver() = delete;

	SuperElasticOrthotropicSolver(
		const int _r,
		const double _p_v,
		const double _alfa,
		const double _a,
		const double _b,
		const double _bulk,
		const double _af,
		const double _bf,
		const double _as,
		const double _bs,
		const double _asf,
		const double _bsf
		// const std::function<FthODTensor(SODTensor&)> _depdef
	) :
		r_deg(_r),
		p_v(_p_v),
		alfa(_alfa),
		a(_a),
		b(_b),
		bulk(_bulk),
		af(_af),
		bf(_bf),
		as(_as),
		bs(_bs),
		asf(_asf),
		bsf(_bsf),
		deP_deF_at_q(dim*dim)
	{ }

	void setup(const std::string& mesh);

	void solve();

	void output() const;

	typedef struct {

		Tensor<2, dim> ss0t;
		Tensor<2, dim> ff0t;
		Tensor<1, dim> s0;
		Tensor<1, dim> f0;
		Tensor<2, dim> Fmt;
		Tensor<2, dim> Finv;

		double i4f;
		double i4s;
		double J;
		double I1;
		double exp_bi1m3;
		double exp_mac_s_sq;
		double exp_mac_f_sq;

	} pass_cache_data_t;


protected:

	void 
	compute_deP_deF_at_q(
		const Tensor<2, dim>& F_q, const pass_cache_data_t& intermediate
	);

	void
	voigt_apply_to(
		const Tensor<2, dim>& at, const Tensor<2, dim>& multiply_by,
		Tensor<2, dim>& into, const pass_cache_data_t&
	);

	void
		voigt_apply_to_batch(
			const Tensor<2, dim>& F, const Tensor <2, dim>& left_hand,
			const std::vector<Tensor<2, dim>>& right_hand,
			const pass_cache_data_t& intermediate,
			Vector<double>& save_into);

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

	void build_system(bool build_jacobian = true);

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
	const unsigned int r_deg;

	const double p_v;

	const double alfa;

	const double a;
	const double b;

	const double bulk;

	const double af;
	const double bf;

	const double as;
	const double bs;

	const double asf;
	const double bsf;

	std::vector<Tensor<2, dim>> deP_deF_at_q;


};



#endif // !__SOLVER_SUPERELASTIC_ISOTROPIC
