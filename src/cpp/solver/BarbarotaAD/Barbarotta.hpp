#ifndef BARBAROTTA_HPP
#define BARBAROTTA_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/physics/elasticity/kinematics.h>

#include <deal.II/lac/affine_constraints.h>

//----------AD header--------
#include <deal.II/differentiation/ad.h>
//---------------------------

#include <fstream>
#include <iostream>
#include <vector>

using namespace dealii;

class BarbarottaSolver {
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  BarbarottaSolver(const unsigned int r_)
      : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
        mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
        pcout(std::cout, mpi_rank == 0), r(r_), mesh(MPI_COMM_WORLD) {}

  void setup(const std::string &mesh_path);

  void solve_newton(const std::string &output_file_name);

protected:
  using ADNumber = typename Differentiation::AD::ResidualLinearization<
      Differentiation::AD::NumberTypes::sacado_dfad, double>::ad_type;


  	typedef struct {

    Tensor<2, dim, ADNumber> Finv, Fmt;
    ADNumber J;

    ADNumber i4s, i4f, i8sf;
   
    ADNumber macf, macs;

    ADNumber exp_mac_s_sq, exp_mac_f_sq, exp_bi8sf_sq, exp_bi1m3;
    
    Tensor<2, dim, ADNumber> f0f0t, s0s0t;
    //Tensor<1, dim, ADNumber> f0, s0;

  } pass_cache_data_t;

  void assemble_system();

  void solve();

  void compute_P_at_q(const Tensor<2, dim, ADNumber> &F_q,
                      const pass_cache_data_t &i);

  ADNumber active_phi(const ADNumber i4);

  //----------------------------MPI-------------------------------------

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  //--------------------------------------------------------------------

  const unsigned int r;

  //----Orthotropic basis-----
  Tensor<1, dim, ADNumber> f0;
  Tensor<1, dim, ADNumber> s0;
  Tensor<1, dim, ADNumber> n0;
  void initialize_orth_basis(const dealii::Point<dim> &p);
  //--------------------------

  //----Barbarotta's parameters----
  const ADNumber alfa = 0.4;
  const ADNumber a = 0.2;
  const ADNumber b = 4.6140;
  const ADNumber af = 4.1907;
  const ADNumber bf = 7.8565;
  const ADNumber as = 2.5640;
  const ADNumber bs = 10.446;
  const ADNumber asf = 0.13040;
  const ADNumber bsf = 15.255;
  ADNumber Sn = 5.1;
  const ADNumber beta = 4.1;
  const ADNumber bulk = 5.0;
  // Pressure for Neumann
  ADNumber p_v = 0.1;
  //--------------------------


  //----Fiber's parameters-----

  const double endo_max = 17.0;
  const double endo_min = 7.0;
  const double epi_max = 20.0;
  const double epi_min = 10.0;

  //---------------------------

  // Piola-Kirchoff
  Tensor<2, dim, ADNumber> P;

  AffineConstraints<double> constraints;


  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // Quadrature formula for face integrals.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Jacobian matrix.
  TrilinosWrappers::SparseMatrix jacobian;

  // Residual vector.
  TrilinosWrappers::MPI::Vector nr_rhs_f;

  // Solution increment (without ghost elements).
  TrilinosWrappers::MPI::Vector step_owned;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;
};

#endif