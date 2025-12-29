#pragma once
#ifndef __SOLVER_SUPERELASTIC_ISOTROPIC
#define __SOLVER_SUPERELASTIC_ISOTROPIC

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>
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


#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <functional>

using namespace dealii;

class OrthotropicSolver {

public:
	static constexpr unsigned int dim = 3;

	// Fourth order tensor acting on dimxdim matrices
	typedef dealii::Tensor<2, dim, double> SODTensor;
	typedef dealii::Tensor<4, dim, double> FthODTensor;

	OrthotropicSolver() = delete;

	OrthotropicSolver(
		const int _r,
		const double _ch_p,
		const double _alfa,
		const double _a,
		const double _b,
		const double _af,
		const double _bf,
		const double _as,
		const double _bs,
		const double _asf,
		const double _bsf,
		const double _Sn,
		const double _beta,
		const double _bulk
	) :
		r_deg(_r),
		p_v(_ch_p),
		alfa(_alfa),
		a(_a),
		b(_b),
		af(_af),
		bf(_bf),
		as(_as),
		bs(_bs),
		asf(_asf),
		bsf(_bsf),
		Sn(_Sn),
		beta(_beta),
		bulk(_bulk),
		mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
		mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
		mesh(MPI_COMM_WORLD),
		pcout(std::cout, mpi_rank == 0), deP_deF_at_q(dim*dim)
	{ }

	void
		setup(const std::string& mesh);

	void
		solve(const std::string&);

	void
		output() const;

protected:

	typedef struct {

		Tensor<2, dim> Finv;
		Tensor<2, dim> Fmt;
		// Isochoric and volumetric terms
		double J;
		double I1;
		double exp_bi1m3;
		// Orthotropic terms
		double i4f;
		double i4s;
		double macs;
		double macf;
		double exp_mac_s_sq;
		double exp_mac_f_sq;
		Tensor<2, dim> f0f0t;
		Tensor<2, dim> s0s0t;
		double i8sf;
		double exp_bi8sf_sq;

		Tensor<1, dim> f0;
		Tensor<1, dim> s0;
		// Active terms

	} pass_cache_data_t;


	void compute_deP_deF_at_q(
		const Tensor<2, dim>& F_q, const pass_cache_data_t& intermediate
	);

	void compute_P_at_q(
		const Tensor<2, dim>& F_q, const pass_cache_data_t& intermediate
	);

	void build_system(bool build_jacobian = true);

	void
		compute_basis_at_quadrature(
			const std::vector<Point<dim>>& p,
			std::vector<std::vector<Tensor<1, dim>>>& orth_sys,
			bool compute_n
		);

	void
		orthothropic_base_at(
			const Point<dim>& p, std::vector<Tensor<1, dim>>& basis, bool compute_n = false
		);

	double compute_internal_volume( );

	double compute_external_volume( );

	double
		active_phi(const double i4);

	double
		active_phi_prime(const double i4);


	const unsigned int r_deg;

	double p_v;

	const double alfa;

	const double a;

	const double b;

	const double af;

	const double bf;

	const double as;

	const double bs;

	const double asf;

	const double bsf;

	const double Sn;

	const double beta; 

	double bulk;
	// Number of MPI processes.
	const unsigned int mpi_size;

	// Rank of the current MPI process.
	const unsigned int mpi_rank;

	// Triangulation.
	parallel::fullydistributed::Triangulation<dim> mesh;

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
	TrilinosWrappers::MPI::Vector solution;

	TrilinosWrappers::MPI::Vector step_owned;

	// System solution (without ghost elements).
	TrilinosWrappers::MPI::Vector solution_owned;

	TrilinosWrappers::MPI::Vector nr_rhs_f;

	TrilinosWrappers::SparseMatrix jacobian;

	dealii::AffineConstraints<double> constraints;

	ConditionalOStream pcout;

	std::vector<Tensor<2, dim>> deP_deF_at_q;
	Tensor<2, dim> P_at_q;

};


#endif // !__SOLVER_SUPERELASTIC_ISOTROPIC
