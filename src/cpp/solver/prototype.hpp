#ifndef NON_LINEAR_ORTH_HPP
#define NON_LINEAR_ORTH_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

class NonLinearOrthotropic
{
public:

    static constexpr unsigned int dim = 3;

    // Constructor.
    NonLinearOrthotropic(const std::string& mesh_file_name_,
        const unsigned int& r_)
        : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank == 0)
        , mesh_file_name(mesh_file_name_)
        , r(r_)
        , mesh(MPI_COMM_WORLD)
    {
    }

    // Initialization.
    void setup();

    // Solve the problem using Newton's method.
    void solve_newton();

    // Output.
    void
        output() const;

protected:

    // Assemble the tangent problem.
    void
        assemble_system(bool assemble_jacobian = true);

    // Solve the tangent problem.
    void
        solve_system();


    const unsigned int mpi_size;

    const unsigned int mpi_rank;

    ConditionalOStream pcout;

    // Mesh file name.
    const std::string mesh_file_name;

    // Polynomial degree.
    const unsigned int r;

    // Mesh.
    parallel::fullydistributed::Triangulation<dim> mesh;

    // Finite element space.
    std::unique_ptr<FiniteElement<dim>> fe;

    // Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature;
    std::unique_ptr<Quadrature<dim-1>> surf_quadrature;

    // DoF handler.
    DoFHandler<dim> dof_handler;

    // DoFs owned by current process.
    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    // Jacobian matrix.
    TrilinosWrappers::SparseMatrix jacobian_matrix;

    // Residual vector.
    TrilinosWrappers::MPI::Vector residual_vector;

    // Solution increment (without ghost elements).
    TrilinosWrappers::MPI::Vector delta_owned;

    // System solution (without ghost elements).
    TrilinosWrappers::MPI::Vector solution_owned;

    // System solution (including ghost elements).
    TrilinosWrappers::MPI::Vector solution;
};

#endif