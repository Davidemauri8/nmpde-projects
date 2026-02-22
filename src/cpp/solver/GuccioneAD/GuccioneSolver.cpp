#include "GuccioneSolver.hpp"
#include "BoundariesGuccione.hpp"

#include <deal.II/base/function.h>
#include <deal.II/differentiation/ad.h>

#include <map>
#include <cmath>
#include "../utilities/mesh_io.hpp"

// Defining the method to populate the struct
#define compute_and_cache(name, expression, inter) const auto name = expression; inter.name = name; 

using ADHelper =
        Differentiation::AD::ResidualLinearization<
        Differentiation::AD::NumberTypes::sacado_dfad,
        double>;
using ADNumber = typename ADHelper::ad_type;

void GuccioneSolver::setup()
{
  // Create the mesh.
  {
    pde_out_c_par(pcout, "Initializing the mesh", RED_COLOR);


    Triangulation<dim> mesh_serial;

    GridGenerator::subdivided_hyper_rectangle(
        mesh_serial, 
        {10, 1, 1},              // Subdivisions per axis
        Point<dim>(0, 0, 0),     // Origin
        Point<dim>(10, 1, 1),    // Opposite corner
        true                     // Colorize boundaries
    );

    // Perform one level of uniform refinement
    mesh_serial.refine_global(1);

    // Then, we copy the triangulation into the parallel one.
    {
      GridTools::partition_triangulation(mpi_size, mesh_serial);
      const auto construction_data = TriangulationDescription::Utilities::
          create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
      mesh.create_triangulation(construction_data);
    }

    // Notice that we write here the number of *global* active cells (across all
    // processes).
    pde_out_c_par(pcout,
                  "  Number of elements = " << mesh.n_global_active_cells(),
                  RED_COLOR);
  }
    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the finite element space. This is the same as in serial codes.
    {
    pde_out_c_par(pcout, "Initializing the finite element space", RED_COLOR);

    FE_Q<dim> fe_scalar(r);
    fe = std::make_unique<FESystem<dim>>(fe_scalar, dim);

    pde_out_c_par(pcout, "  Degree                     = " << fe->degree,
                  RED_COLOR);
    pde_out_c_par(pcout, "  DoFs per cell              = " << fe->dofs_per_cell,
                  RED_COLOR);

    quadrature = std::make_unique<QGauss<dim>>(r + 1);

    pde_out_c_par(pcout,
                  "  Quadrature points per cell = " << quadrature->size(),
                  RED_COLOR);

    quadrature_face = std::make_unique<QGauss<dim - 1>>(fe->degree + 1);

    pde_out_c_par(pcout,
                  "  Quadrature points per face = " << quadrature_face->size(),
                  RED_COLOR);
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the DoF handler.
    {
    pde_out_c_par(pcout, "Initializing the DoF handler", RED_COLOR);

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // We retrieve the set of locally owned DoFs, which will be useful when
    // initializing linear algebra classes.
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);

    pde_out_c_par(pcout, "  Number of DoFs = " << dof_handler.n_dofs(),
                  RED_COLOR);
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the linear system.
    {
    pde_out_c_par(pcout, "Initializing the linear system", RED_COLOR);

    pde_out_c_par(pcout, "  Initializing the sparsity pattern", RED_COLOR);

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);

    sparsity.compress();

    pde_out_c_par(pcout, "  Initializing the system matrix", RED_COLOR);
    jacobian.reinit(sparsity);

    pde_out_c_par(pcout, "  Initializing the system right-hand side",
                  RED_COLOR);
    nr_rhs_f.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    pde_out_c_par(pcout, "  Initializing the solution vector", RED_COLOR);
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    step_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    }
}


void GuccioneSolver::compute_P_at_q(const Tensor<2, dim, ADNumber> &F_q,
                                       const pass_cache_data_t &i) {

    //--------------------------computing e^Q---------------------------------
    ADNumber normal = bn0n0 * (i.En0n0 * i.En0n0);
    ADNumber ff = bf0f0 * (i.Ef0f0 * i.Ef0f0);
    ADNumber ss = bs0s0 *(i.Es0s0 * i.Es0s0);
    ADNumber mutual = bf0s0 * (i.Ef0s0 * i.Ef0s0 + i.Es0f0 * i.Es0f0) +
                      bf0n0 * (i.Ef0n0 * i.Ef0n0 + i.En0f0 * i.En0f0) +
                      bs0n0 * (i.Es0n0 * i.Es0n0 + i.En0s0 * i.En0s0);

    ADNumber Q = normal + ff + ss + mutual;
    ADNumber expQ = exp(Q);
    //------------------------------------------------------------------------

    //--------------------------computing dQdE-------------------------------
    
    Tensor<2, dim, ADNumber> dQdE;
    dQdE = 2.0 * bf0f0 * i.Ef0f0 * i.f0f0 +
           2.0 * bs0s0 * i.Es0s0 * i.s0s0 +
           2.0 * bn0n0 * i.En0n0 * i.n0n0 + 
           2.0 * bf0s0 * i.Ef0s0 * (i.f0s0 + i.s0f0) +
           2.0 * bf0n0 * i.Ef0n0 * (i.f0n0 + i.n0f0) +
           2.0 * bs0n0 * i.Es0n0 * (i.s0n0 + i.n0s0);
    //------------------------------------------------------------------------


    P = F_q * 0.5 * C * expQ * dQdE;

}



void GuccioneSolver::assemble_system()
{
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();
    const unsigned int n_q_face = quadrature_face->size();

    FEValues<dim> fe_values(*fe,
        *quadrature,
        update_values | update_gradients |
        update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values(*fe, *quadrature_face,
                                     update_values | update_gradients |
                                         update_normal_vectors |
                                         update_JxW_values);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    jacobian = 0.0;
    nr_rhs_f = 0.0;

    //alignas(cache_line_size()) it does not work, how to solve??
    pass_cache_data_t intermediate{};

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) {
            continue;
        }
            

        fe_values.reinit(cell);
        cell->get_dof_indices(dof_indices);

        cell_matrix = 0.0;
        cell_rhs = 0.0;

        FEValuesExtractors::Vector displacement(0);

        const unsigned int n_independent_variables = dof_indices.size();
        const unsigned int n_dependent_variables = dofs_per_cell;
        ADHelper ad_helper(n_independent_variables, n_dependent_variables);

        ad_helper.register_dof_values(solution, dof_indices);

        //DoFs values in auto-differentiable numbers.
        const std::vector<ADNumber>& dof_values_ad =
            ad_helper.get_sensitive_dof_values();
       

        // Problem specific task, compute values and gradients
        std::vector<Tensor<2, dim, ADNumber>> solution_gradient_loc(
            n_q, Tensor<2, dim, ADNumber>());

        fe_values[displacement].get_function_gradients_from_local_dof_values(
            dof_values_ad, solution_gradient_loc);

        // 4) Build AD residual vector (must be explicitly zero-initialized)
        std::vector<ADNumber> residual_ad(n_dependent_variables, ADNumber(0.0));

        for (unsigned int q = 0; q < n_q; ++q){

            // Initialize the orthonormal basis
            initialize_orth_basis();

            // Computing F and E
            const Tensor<2, dim, ADNumber> F = Physics::Elasticity::Kinematics::F(solution_gradient_loc[q]);
            const Tensor<2, dim, ADNumber> E = Physics::Elasticity::Kinematics::E(F);

            //-------------Computing and caching the elements for computing e^Q---------------
            compute_and_cache(Ef0f0, scalar_product(f0, E * f0), intermediate);
            compute_and_cache(Ef0s0, scalar_product(f0, E * s0), intermediate);
            compute_and_cache(Ef0n0, scalar_product(f0, E * n0), intermediate);
            compute_and_cache(Es0s0, scalar_product(s0, E * s0), intermediate);
            compute_and_cache(Es0f0, scalar_product(s0, E * f0), intermediate);
            compute_and_cache(Es0n0, scalar_product(s0, E * n0), intermediate);
            compute_and_cache(En0n0, scalar_product(n0, E * n0), intermediate);
            compute_and_cache(En0f0, scalar_product(n0, E * f0), intermediate);
            compute_and_cache(En0s0, scalar_product(n0, E * s0), intermediate);

            compute_and_cache(f0f0, outer_product(f0, f0), intermediate);
            compute_and_cache(f0s0, outer_product(f0, s0), intermediate);
            compute_and_cache(f0n0, outer_product(f0, n0), intermediate);
            compute_and_cache(s0s0, outer_product(s0, s0), intermediate);
            compute_and_cache(s0n0, outer_product(s0, n0), intermediate);
            compute_and_cache(s0f0, outer_product(s0, f0), intermediate);
            compute_and_cache(n0n0, outer_product(n0, n0), intermediate);
            compute_and_cache(n0f0, outer_product(n0, f0), intermediate);
            compute_and_cache(n0s0, outer_product(n0, s0), intermediate);
            //--------------------------------------------------------------------------------


            compute_P_at_q(F, intermediate);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                residual_ad[i] +=
                    (scalar_product(P, fe_values[displacement].gradient(i, q))) *
                    fe_values.JxW(q);
            }
        }

        // Neumann
        if (cell->at_boundary())
        {
            for (unsigned int f = 0; f < cell->n_faces(); ++f)
            {
                int id = cell->face(f)->boundary_id();
                if (cell->face(f)->at_boundary() && is_neumann(id))
                {
                    fe_face_values.reinit(cell, f);
                  //------------------------------Neumann boundary conditions--------------------------------------------------

                  std::vector<Tensor<2, dim, ADNumber>> gradient_bound(
                        n_q_face,
                                                    Tensor<2, dim, ADNumber>());
                  fe_face_values[displacement]
                      .get_function_gradients_from_local_dof_values(
                          dof_values_ad, gradient_bound);

                  for (unsigned int q = 0; q < n_q_face; ++q) {

                    // Computing F
                    const Tensor<2, dim, ADNumber> F =
                        Physics::Elasticity::Kinematics::F(gradient_bound[q]);
                    // Computing J
                    ADNumber detF = determinant(F);
                    // Computing F^-T
                    Tensor<2, dim, ADNumber> Fmt = invert(transpose(F));
                    // Computing cofactor
                    Tensor<2, dim, ADNumber> cof = detF * Fmt;

                    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                      Tensor<1, dim, ADNumber> n =
                          fe_face_values.normal_vector(q);
                      residual_ad[i] +=
                          p_v *
                          scalar_product(
                              cof * n,
                              fe_face_values[displacement].value(i, q)) *
                          fe_face_values.JxW(q);
                    }
                  }
                }
            }
        }

        ad_helper.register_residual_vector(residual_ad);
        ad_helper.compute_residual(cell_rhs);
        cell_rhs *= -1.0; 
        ad_helper.compute_linearization(cell_matrix);
        jacobian.add(dof_indices, cell_matrix);
        nr_rhs_f.add(dof_indices, cell_rhs);
    }

    jacobian.compress(VectorOperation::add);
    nr_rhs_f.compress(VectorOperation::add);

    // Dirichlet
    {
        std::map<types::global_dof_index, double> boundary_values;
        std::map<types::boundary_id, const Function<dim>*> boundary_functions;
        Functions::ZeroFunction<dim> bc_func;
        boundary_functions[PDE_DIRICHLET] = &bc_func;

        VectorTools::interpolate_boundary_values(dof_handler,
            boundary_functions,
            boundary_values);
        MatrixTools::apply_boundary_values(
            boundary_values, jacobian, step_owned, nr_rhs_f, true);
    }
}

void GuccioneSolver::initialize_orth_basis() {
    // In this case we just need a simple (1,0,0), (0,1,0) and (0,0,1)
    // because explicitly mentioned in https://pmc.ncbi.nlm.nih.gov/articles/PMC4707707/pdf/rspa20150641.pdf

    // Fiber
    f0[0] = 1;  f0[1] = 0;  f0[2] = 0;

    // Strain
    s0[0] = 0;  s0[1] = 1;  s0[2] = 0;

    // Normal
    n0[0] = 0;  n0[1] = 0;  n0[2] = 1;

}

void GuccioneSolver::solve() {
    SolverControl solver_control(1000, 1e-9 * nr_rhs_f.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

    TrilinosWrappers::PreconditionAMG preconditioner;
    TrilinosWrappers::PreconditionAMG::AdditionalData data;

    data.smoother_sweeps = 3;

    if (r > 1) {
        data.higher_order_elements = true;
    } else {
        data.higher_order_elements = false;
    }
    data.elliptic = true;

    preconditioner.initialize(jacobian, data);

    solver.solve(jacobian, step_owned, nr_rhs_f, preconditioner);

    pde_out_c_par(pcout,
                  "   " << solver_control.last_step() << " GMRES iterations",
                  RED_COLOR);
}

void GuccioneSolver::solve_newton(const std::string &output_file_name) {
    pcout << "===============================================" << std::endl;
    #define MAX_ITER_AMT 80

    const double toll = 1e-6;
    unsigned int n_iter = 0;
    double rhs_norm = toll + 1;

    while (n_iter < MAX_ITER_AMT && rhs_norm > toll) {
        assemble_system();
        rhs_norm = nr_rhs_f.l2_norm();

        pde_out_c_par(pcout,
                      "Newton iteration " << n_iter << "/" << MAX_ITER_AMT,
                      RED_COLOR);

        pde_out_c_par(pcout,
                      "The norm of the residual is: " << std::scientific
                                                      << std::setprecision(6)
                                                      << rhs_norm,
                      RED_COLOR);

        if (rhs_norm < toll) {
            n_iter = MAX_ITER_AMT;
            solution = solution_owned;
        } else {
            solve();
            solution_owned += step_owned;
            solution = solution_owned;
        }

        ++n_iter;
    }

    pcout << std::endl;
    pcout << "===============================================" << std::endl;

    //------------------------------Output--------------------------------------
    pde_out_c_par(pcout, "Completed the newton iteration, saving the result",
                  GRN_COLOR);

    DataOut<dim> data_out;

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_interpretation(
            dim, DataComponentInterpretation::component_is_part_of_vector);
    std::vector<std::string> name(dim, "solution");

    data_out.add_data_vector(dof_handler, solution, name, data_interpretation);
    // Change into this to plot just the z value
    //data_out.add_data_vector(dof_handler, solution, "solution");

    std::vector<unsigned int> partition_int(mesh.n_active_cells());
    GridTools::get_subdomain_association(mesh, partition_int);
    const Vector<double> partitioning(partition_int.begin(),
                                      partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record("./", output_file_name, 0,
                                        MPI_COMM_WORLD);

    pde_out_c_par(pcout, "Result saved into file " << output_file_name,
                  GRN_COLOR);
    //-------------------------------------------------------------------------

}


