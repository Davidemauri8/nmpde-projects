#ifndef HEART_GUCCIONE_HPP
#define HEART_GUCCIONE_HPP

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

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/physics/elasticity/kinematics.h>
//----------AD header--------
#include <deal.II/differentiation/ad.h>
//---------------------------

#include <fstream>
#include <iostream>
#include <vector>

using namespace dealii;

class GuccioneSolver {
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  GuccioneSolver(const unsigned int r_)
      : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
        mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
        pcout(std::cout, mpi_rank == 0), r(r_), mesh(MPI_COMM_WORLD) {}

  void setup(const std::string &mesh_path);

  void solve_newton(const std::string &output_file_name);

protected:
  using ADNumber = typename Differentiation::AD::ResidualLinearization<
      Differentiation::AD::NumberTypes::sacado_dfad, double>::ad_type;

  // A struct used to pass data around the code efficiently
  typedef struct {

    ADNumber Ef0f0, Ef0s0, Ef0n0;
    ADNumber Es0s0, Es0n0, Es0f0;
    ADNumber En0n0, En0f0, En0s0;

    Tensor<2, dim, ADNumber> f0f0, f0s0, f0n0;
    Tensor<2, dim, ADNumber> s0s0, s0n0, s0f0;
    Tensor<2, dim, ADNumber> n0n0, n0f0, n0s0;

  } pass_cache_data_t;

  void assemble_system();

  void solve();

  void compute_P_at_q(const Tensor<2, dim, ADNumber> &F_q,
                      const pass_cache_data_t &i);

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
  void initialize_orth_basis();
  //--------------------------

  //----Guccione's parameters----
  // Parameters taken from :
  // https://pmc.ncbi.nlm.nih.gov/articles/PMC4707707/pdf/rspa20150641.pdf
  const double bf0f0 = 1;
  const double bf0s0 = 1;
  const double bf0n0 = 1;
  const double bs0s0 = 1;
  const double bs0n0 = 1;
  const double bn0n0 = 1;
  const double C = 10000;
  // Pressure for Neumann
  ADNumber p_v = 5000;
  //--------------------------

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