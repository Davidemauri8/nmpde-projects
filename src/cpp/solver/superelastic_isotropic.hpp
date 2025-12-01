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

class SuperElasticIsotropicSolver {


public:
	static constexpr unsigned int dim = 3;

	// Fourth order tensor acting on dimxdim matrices
	typedef dealii::Tensor<2, dim, double> SODTensor;
	typedef dealii::Tensor<4, dim, double> FthODTensor;

	SuperElasticIsotropicSolver() = delete;

	SuperElasticIsotropicSolver(
		const int _r,
		const double _ch_p,
		const double _alfa,
		const double _mu,
		const double _bulk,
		const double _af,
		const double _as
		// const std::function<FthODTensor(SODTensor&)> _depdef
	) :
		r(_r),
		ch_p(_ch_p),
		alfa(_alfa), 
		mu(_mu),
		bulk(_bulk),
		af(_af),
		as(_as)
	{ }

	void
		setup(const std::string& mesh);

	void
		solve();

	void
		output() const;

protected:

	dealii::Tensor<2, dim> voigt_apply_to(
		const dealii::Tensor<2,dim>&, const dealii::Tensor<2, dim>&);


	void orthothropic_base_at(
		const dealii::Point<dim>& p, std::vector<dealii::Tensor<1, dim>>& basis,
		bool compute_n
	);

	void
		compute_basis_at_quadrature(
			const std::vector<dealii::Point<dim>>& p,
			std::vector<std::vector<dealii::Tensor<1, dim>>>& orth_sys,
			bool compute_n
		);

	void compute_rh_s_newt_raphs(dealii::Vector<double>& put);

	void build_jacobian();

	// Triangulation.
	dealii::Triangulation<dim> mesh;

	// Finite element space.
	std::unique_ptr<dealii::FiniteElement<dim>> fe;

	// Quadrature formula.
	std::unique_ptr<dealii::Quadrature<dim>> quadrature;
	
	std::unique_ptr<dealii::Quadrature<dim - 1>> surf_quadrature;

	// DoF handler.
	dealii::DoFHandler<dim> dof_handler;

	// Sparsity pattern.
	dealii::SparsityPattern sparsity_pattern;

	// System solution.
	dealii::Vector<double> solution;

	dealii::Vector<double> step;


	dealii::Vector<double> nr_rhs_f;

	dealii::SparseMatrix<double> jacobian;

	dealii::AffineConstraints<double> constraints;

	// dealii::AffineConstraints<double> constraints;

	// Polynomial degree.
	const unsigned int r;

	const double ch_p;

	const double alfa;

	const double mu;

	const double bulk;
	
	const double af;

	const double as;

	const std::function<FthODTensor(const SODTensor&)> depdef;



};
	

#endif // !__SOLVER_SUPERELASTIC_ISOTROPIC
