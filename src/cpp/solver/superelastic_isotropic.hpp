#pragma once
#ifndef __SOLVER_SUPERELASTIC_ISOTROPIC
#define __SOLVER_SUPERELASTIC_ISOTROPIC

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

class SuperElasticIsotropicSolver {

	static constexpr unsigned int dim = 3;

public:

	SuperElasticIsotropicSolver() = delete;

	SuperElasticIsotropicSolver(
		const int _r,
		const double _ch_p,
		const bool _do_load_stepping = true
	) : 
		r(_r),
		ch_p(_ch_p),
		do_load_stepping(_do_load_stepping)
	{ }

	void
	setup(const std::string& mesh);

	void 
	solve();
	
	void
	output() const;

protected:

	// Triangulation.
	dealii::Triangulation<dim> mesh;

	// Finite element space.
	std::unique_ptr<dealii::FiniteElement<dim>> fe;

	// Quadrature formula.
	std::unique_ptr<dealii::Quadrature<dim>> quadrature;

	// DoF handler.
	dealii::DoFHandler<dim> dof_handler;

	// Sparsity pattern.
	dealii::SparsityPattern sparsity_pattern;

	// System solution.
	dealii::Vector<double> solution;

	dealii::SparseMatrix<double> jacobian;
	// Polynomial degree.
	const unsigned int r;

	double ch_p;

	bool do_load_stepping;

};
	

#endif // !__SOLVER_SUPERELASTIC_ISOTROPIC
