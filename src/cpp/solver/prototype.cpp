#include "rebuild_new.hpp"


void
NonLinearOrthotropic::setup() 
{
    // Create the mesh.
    {
        pcout << "Initializing the mesh" << std::endl;

        // First we read the mesh from file into a serial (i.e. not parallel)
        // triangulation.
        Triangulation<dim> mesh_serial;
        UtilsMesh::load_mesh_into_tria(mesh_file_name, mesh_serial);
        {
            GridTools::partition_triangulation(mpi_size, mesh_serial);
            const auto construction_data = TriangulationDescription::Utilities::
                create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
            mesh.create_triangulation(construction_data);
        }

        pcout << "  Number of elements = " << mesh.n_global_active_cells()
            << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the finite element space. This is the same as in serial codes.
    {
        pcout << "Initializing the finite element space" << std::endl;

        fe = std::make_unique<FE_SimplexP<dim>>(r);

        pcout << "  Degree                     = " << fe->degree << std::endl;
        pcout << "  DoFs per cell              = " << fe->dofs_per_cell
            << std::endl;

        quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);
        surf_quadrature = std::make_unique<QGaussSimplex<dim-1>>(r + 1);

        pcout << "  Quadrature points per cell = " << quadrature->size()
            << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the DoF handler.
    {
        pcout << "Initializing the DoF handler" << std::endl;

        dof_handler.reinit(mesh);
        dof_handler.distribute_dofs(*fe);

        // We retrieve the set of locally owned DoFs, which will be useful when
        // initializing linear algebra classes.
        locally_owned_dofs = dof_handler.locally_owned_dofs();
        locally_relevant_dofs =
            DoFTools::extract_locally_relevant_dofs(dof_handler);

        pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the linear system.
    {
        pcout << "Initializing the linear system" << std::endl;

        pcout << "  Initializing the sparsity pattern" << std::endl;

        // To initialize the sparsity pattern, we use Trilinos' class, that manages
        // some of the inter-process communication.
        TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
            MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler, sparsity);

        // After initialization, we need to call compress, so that all process
        // retrieve the information they need for the rows they own (i.e. the rows
        // corresponding to locally owned DoFs).
        sparsity.compress();

        // Then, we use the sparsity pattern to initialize the system matrix. Since
        // the sparsity pattern is partitioned by row, so will the matrix.
        pcout << "  Initializing the system matrix" << std::endl;
        jacobian_matrix.reinit(sparsity);

        // Finally, we initialize the right-hand side and solution vectors.
        pcout << "  Initializing the system right-hand side" << std::endl;
        residual_vector.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        pcout << "  Initializing the solution vector" << std::endl;
        solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
        delta_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    }
}