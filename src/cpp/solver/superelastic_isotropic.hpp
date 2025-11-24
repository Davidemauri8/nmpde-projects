#pragma once
#ifndef __SOLVER_SUPERELASTIC_ISOTROPIC
#define __SOLVER_SUPERELASTIC_ISOTROPIC

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <functional>

class SuperElasticIsotropicSolver {

	static constexpr unsigned int dim = 3;

public:

	// Fourth order tensor acting on dimxdim matrices
	typedef dealii::Tensor<2, dim, double> FODTensor;

	SuperElasticIsotropicSolver() = delete;

	SuperElasticIsotropicSolver(
		const int _r,
		const double _ch_p,
		const double _alfa,
		const std::function<void(FODTensor&)> _depdef
	) : 
		r(_r),
		ch_p(_ch_p),
		alfa(_alfa),
		depdef(_depdef)
	{ }

	void
			setup(const std::string& mesh);

	void 
		solve();
	
	void
		output() const;

protected:

	void
		build_jacobian();

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

	dealii::Vector<double> nr_rhs_f;

	dealii::SparseMatrix<double> jacobian;

	// Polynomial degree.
	const unsigned int r;

	const double ch_p;

	const double alfa;

	const std::function<void(FODTensor&)> depdef;

};
	

#endif // !__SOLVER_SUPERELASTIC_ISOTROPIC
