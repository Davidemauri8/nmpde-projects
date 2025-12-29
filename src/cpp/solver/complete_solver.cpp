#include <cmath>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>


#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/lac/sparse_ilu.h>
#include <fstream>
#include <iostream>
#include <functional>


#include "complete_solver.hpp"
#include "mesh_geometry.hpp"
#include "boundaries.hpp"

// Uncomment this definition to get verbose output (datetime and code line)
#define PDE_OUT_VERBOSE
#define PDE_PROGRESS_BAR
#include "../utilities/visualize.hpp"
#include "../utilities/mesh_io.hpp"
#include "../utilities/math.hpp"

#define key(s1, s) s


using namespace dealii;

void
OrthotropicSolver::setup(
    const std::string& mesh_path
) {
    Triangulation<dim> mesh_serial;
    {
        pde_out_c_par(pcout, "Loading mesh " << mesh_path, RED_COLOR);
        UtilsMesh::load_mesh_into_tria(mesh_path, mesh_serial);

    }
    {
        GridTools::partition_triangulation(mpi_size, mesh_serial);

        const auto construction_data = TriangulationDescription::Utilities::
            create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
        mesh.create_triangulation(construction_data);
    }
    {
        pde_out_c_par(pcout, "Initializing the finite element space", YEL_COLOR);

        fe = std::make_unique<FESystem<dim>>(FE_SimplexP<dim>(this->r_deg) ^ dim);

        pde_out_c_par(pcout, "Degree = " << fe->degree, YEL_COLOR);
        pde_out_c_par(pcout, "DoFs per cell = " << fe->dofs_per_cell, YEL_COLOR);

        quadrature = std::make_unique<QGaussSimplex<dim>>(r_deg + 1);
        surf_quadrature = std::make_unique<QGaussSimplex<dim - 1>>(r_deg + 1);

        pde_out_c_par(pcout, "Volume quadrature points per cell = " << quadrature->size(), YEL_COLOR);
        pde_out_c_par(pcout, "Surface quadrature points per cell = " << surf_quadrature->size(), YEL_COLOR);

    }
    {
        pde_out_c_par(pcout, "Initializing the DoF handler", BLU_COLOR);

        dof_handler.reinit(mesh);
        dof_handler.distribute_dofs(*fe);

        pde_out_c_par(pcout, "Number of DoFs = " << dof_handler.n_dofs(), BLU_COLOR);
    }


    const IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();
    const IndexSet &locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);

    {

        // constraints.clear();
        // DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        /*
        FEValuesExtractors::Scalar z_axis(2);
        ComponentMask z_axis_mask = fe->component_mask(z_axis);

        VectorTools::interpolate_boundary_values(
            dof_handler,
            types::boundary_id(PDE_DIRICHLET),
            Functions::ZeroFunction<dim>(dim),
            constraints,
            z_axis_mask);
        */
        // constraints.close();

        pde_out_c_par(pcout, "Initializing the sparsity pattern", GRN_COLOR);

        TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
            MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler, sparsity
            , constraints, true);
        sparsity.compress();

        pde_out_c_par(pcout, "Copying the sparsity pattern", GRN_COLOR);

        jacobian.reinit(sparsity);

        pde_out_c_par(pcout, "Initializing the solution vector", GRN_COLOR);

        nr_rhs_f.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
        step_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);


    }

    {
        constraints.clear();
        constraints.reinit(locally_relevant_dofs);
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);

        FEValuesExtractors::Scalar z_axis(2);
        ComponentMask z_axis_mask = fe->component_mask(z_axis);

        VectorTools::interpolate_boundary_values(
            dof_handler,
            types::boundary_id(PDE_DIRICHLET),
            Functions::ZeroFunction<dim>(dim),
            constraints,
            z_axis_mask);

        constraints.close();
    }

    solution = 0.0;

}

constexpr const auto dim = OrthotropicSolver::dim;

void
OrthotropicSolver::solve(const std::string& output_file_name) {

    const unsigned int dofs_per_cell = fe->dofs_per_cell;

    pde_out_c_par(pcout, "Solving the non linear mechanics: ", RED_COLOR);
    pde_out_c_par(pcout, "DOFS per Cell " << dofs_per_cell, RED_COLOR);

#define MAX_ITER_AMT 25
    //     solution = 0.0;

    ReductionControl solver_control(
        key("Max iterations", 5000),
        key("Tolerance", 1.0e-9)
    );
    // We use GMRES as the matrix is in general non symmetric. 
    SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

    for (int LOAD_STEPPING = 0; LOAD_STEPPING < 7; LOAD_STEPPING++) {
        // TODO: (just a little thing) add a break when a given threshold residual
        // is achieved at each load_stepping step!
        for (unsigned int newton_iter = 0; newton_iter < MAX_ITER_AMT; ++newton_iter) {

            const double alpha_k = 0.15 + 0.20 * (newton_iter) / MAX_ITER_AMT;

            pde_out_c_par(pcout, "Entering step number: " << newton_iter, YEL_COLOR);
            pde_out_c_par(pcout, "Assembling the linear system", RED_COLOR);

            this->build_system();

            pde_out_c_par(pcout, "Solving the linear system:", RED_COLOR);
            pde_out_c_par(pcout, "L2 Norm of residual: " << nr_rhs_f.l2_norm(), RED_COLOR);
            pde_out_c_par(pcout, "L2 Norm of Jacobian: " << jacobian.l1_norm(), RED_COLOR);

            /*  ---- Notice ----
            * When using a non object oriented constraint (e.g. not just the z component,
            * uncomment this section of the code */
            /* MatrixTools::apply_boundary_values(
                boundary_values, jacobian, step, nr_rhs_f, false
            ); */

            TrilinosWrappers::PreconditionAMG precondition_amg;
            precondition_amg.initialize(jacobian);


            try {
                solver.solve(jacobian, step_owned, nr_rhs_f, precondition_amg);
            }
            catch (const SolverControl::NoConvergence& e) {
                pde_out_c_par(pcout, "FAILED to solve the GMRES iteration at step " << newton_iter, RED_COLOR);
                // pde_out_c("Quitting the iteration (the partial results will be written). "
                //    << newton_iter, RED_COLOR);
                pde_out_c_par(pcout, "Originally: " << e.what(), RED_COLOR);
                break;
            }
            pde_out_c_par(pcout, "Complete after " << solver_control.last_step() << " GMRES iterations", RED_COLOR);
            pde_out_c_par(pcout, "L2 Norm of the step: " << step_owned.l2_norm(), RED_COLOR);

            solution_owned.add(-alpha_k, step_owned);
            // A note: in dealii programs the constraint is distributed to the 
            // entire solution, while in 
            // https://dealii.org/current/doxygen/deal.II/step_32.html 
            // only to the local contribution. We choose the latter approach
            // constraints.distribute(solution);
            constraints.distribute(solution_owned);

            solution = solution_owned;
            // "false" newton iteration keeping the jacobian fixed, has trivial 
            // computational cost and aids convergence
            for (int NR = 0; NR < 4; ++NR) {
                this->build_system(false);
                solver.solve(jacobian, step_owned, nr_rhs_f, precondition_amg);
                solution_owned.add(-alpha_k/3, step_owned);
                constraints.distribute(solution_owned);
                solution = solution_owned;
                pde_out_c_par(pcout, "New iteration residual: " << nr_rhs_f.l2_norm() , RED_COLOR);
            }

            /*
            *  This was a previous implementation of line search for the single threaded
            * implementation of deal-ii.
            * TODO: adapt this to MPI using distributed vectors and the line search utils
            * from dealii.
            * 
            constexpr const double factor = 2;
            while (nr_rhs_f.l2_norm() > cur_norm && lambda > (1 / 128.0)) {
                pde_out_c("Current res " << nr_rhs_f.l2_norm() << " previous " << cur_norm, BLU_COLOR);
                solution = line_search_temp;
                lambda /= factor;
                solution.add(-lambda, step);
                this->build_system(false);
                pde_out_c("Value of " << lambda * 1.5 << " failed for lambda, attempting " << lambda, RED_COLOR);
            } 
            if (cur_norm < nr_rhs_f.l2_norm()) {
                solution = line_search_temp;
                this->build_system(false);
                // The update has failed, switch to a gradient descent iteration on the
                // sum of squared of the residual.
                jacobian.Tvmult(step, nr_rhs_f);
                lambda = 0.1;
                solution.add(-lambda, step);

                this->build_system(false);
                constexpr const double factor = 2;
                while (nr_rhs_f.l2_norm() > cur_norm && lambda > (1 / 256.0)) {
                    pde_out_c("Steepest current res " << nr_rhs_f.l2_norm() << " previous " << cur_norm, BLU_COLOR);
                    solution = line_search_temp;
                    lambda /= factor;
                    solution.add(-lambda, step);
                    this->build_system(false);
                    pde_out_c("Steepest value of " << lambda * 1.5 << " failed for lambda, attempting " << lambda, RED_COLOR);
                }
            }*/

            // At each step, ensure the solution has the right constraints. Use the
            // object oriented version to just limit the z component of the displacement
            // constraints.distribute(solution);

        }
        // Incomplete placeholder for pressure load stepping: 
        // just sub with max_press - init_press / It_amt
        p_v += 0.10;
        pde_out_c_par(pcout, "Repeating the stepping iteration.", RED_COLOR);
    }

    pde_out_c_par(pcout, "Completed the newton iteration, saving the result", GRN_COLOR);
    DataOut<dim> data_out;

    data_out.add_data_vector(dof_handler, solution, "solution");

    std::vector<unsigned int> partition_int(mesh.n_active_cells());
    GridTools::get_subdomain_association(mesh, partition_int);
    const Vector<double> partitioning(partition_int.begin(), partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record("./",
        output_file_name, 0,
        MPI_COMM_WORLD);

    pde_out_c_par(pcout, "Result saved into file " << output_file_name, GRN_COLOR);

    return;
}

#define compute_and_cache(name, expression, inter) const auto name = expression; inter.name = name; 

double
OrthotropicSolver::active_phi(const double i4) {
    return Sn * (1 + beta * (std::sqrt(i4) - 1.0)) / (i4 * std::sqrt(i4));
}

double
OrthotropicSolver::active_phi_prime(const double i4) {
    const auto i452 = std::pow(i4, 5.0 / 2);
    return (-3 / 2.0) * Sn * (1 + beta * (std::sqrt(i4) - 1)) / (i452)+
        Sn * beta / (2 * i4*i4);
}


void OrthotropicSolver::compute_deP_deF_at_q(
    const Tensor<2, dim>& F_q, const pass_cache_data_t& D
) {
    Tensor<2, dim> r, m, f;
    const auto Ff0f0t = F_q * D.f0f0t;
    const auto Fs0s0t = F_q * D.s0s0t;
    const auto fs0_plus_sf0 = outer_product(F_q * D.f0, D.s0) + outer_product(F_q * D.s0, D.f0);
    const auto mixed = outer_product(D.f0, D.s0) + outer_product(D.s0, D.f0);
    const auto f_mix = F_q * mixed;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {

            // Compute the partial derivative of F^-t with respect to F_ij
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    // Notice: we transpose in place by constructing into m[l][k] instead of
                    // m[k][l]
                    m[l][k] = -D.Finv[k][i] * D.Finv[j][l];

            // Contribution of bulk factor
            r = 0.0;
            r += (bulk)*D.J * D.J * D.Fmt * D.Fmt[i][j];
            r += (bulk / 2) * (D.J * D.J - 1) * m;

            // Contribution from the isotropic nonlinear term
            const double nu = 2 * pow_m2t(D.J) * (F_q[i][j] - (1 / 3.0) * D.Fmt[i][j] * D.I1);
            f = (F_q - (1 / 3.0) * D.Fmt) * (b * nu - (2/3.0) *D.Fmt[i][j]);
            f[i][j] += 1;
            f -= (1 / 3.0) * m;
            f *= a * pow_m2t(D.J) * D.exp_bi1m3;

            r += f;

            // Contribution from the fibers
            if (D.macf > 0) {
                r += 4 * af * Ff0f0t[i][j] * D.exp_mac_f_sq * Ff0f0t;
                r += 8 * af * D.macf * D.macf * bf * Ff0f0t[i][j] * D.exp_mac_f_sq * Ff0f0t;
                f = 0.0;
                f[i][j] = 1.0;
                r += 2 * af * D.macf * D.exp_mac_f_sq * f * D.f0f0t;
            }
            // Contribution from the collagen sheets
            if (D.macs > 0) {
                r += 4 * as * Fs0s0t[i][j] * D.exp_mac_s_sq * Fs0s0t;
                r += 8 * as * D.macs * D.macs * bs * Fs0s0t[i][j] * D.exp_mac_s_sq * Fs0s0t;
                f = 0.0;
                f[i][j] = 1.0;
                r += 2 * as * D.macs * D.exp_mac_s_sq * f * D.s0s0t;
            }

            // Cross term for the fibers
            f = 0.0;
            f[i][j] = 1.0;
            r += asf* fs0_plus_sf0 * 2 * f_mix[i][j] * D.exp_bi8sf_sq * (1 + 2 * bsf * D.i8sf * D.i8sf);
            r += asf * D.i8sf * D.exp_bi8sf_sq * f * mixed;

            // Active term
            r += active_phi_prime(D.i4f) * Ff0f0t[i][j] * Ff0f0t + active_phi(D.i4f) * f * D.f0f0t;

            for (int l = 0; l < 3; ++l)
                for (int k = 0; k < 3; ++k)
                    deP_deF_at_q[l*3+k][i][j] = r[l][k];
        }
    }
}

void OrthotropicSolver::compute_P_at_q(
    const Tensor<2, dim>& F_q, const pass_cache_data_t& i
) {

    const auto Ff0f0t = F_q * i.f0f0t;
    const auto Fs0s0t = F_q * i.s0s0t;
    const auto fs0_plus_sf0 = outer_product(F_q * i.f0, i.s0) + outer_product(F_q * i.s0, i.f0);

    // P_at_q = a * pow_m2t(i.J) * (F_q - (1 / 3.0) * i.Fmt);
    P_at_q = (bulk / 2) * (i.J * i.J - 1) * i.Fmt;
    const auto e = a * pow_m2t(i.J) * i.exp_bi1m3 * (F_q - (1 / 3.0) * i.Fmt);
    P_at_q += e;

    // Now, the contribution from orthotropic terms
    // pde_out_c_par(pcout, " R " << e, BLU_COLOR);
    if (i.macf > 0) {
        P_at_q += 2 * af * i.macf * i.exp_mac_f_sq * Ff0f0t;
    }
    if (i.macs > 0) {
        P_at_q += 2 * as * i.macs * i.exp_mac_s_sq * Fs0s0t;
    }
    P_at_q += asf * i.i8sf * i.exp_bi8sf_sq * fs0_plus_sf0;
    // Finally the active stress contribution
    // P_at_q += b * i.J * i.Fmt;
    P_at_q += Ff0f0t * active_phi(i.i4f);

}

void
OrthotropicSolver::build_system(bool build_jacobian) {


    // Number of local DoFs for each element.
    const unsigned int dofs_per_cell = fe->dofs_per_cell;

    // Number of quadrature points for each element.
    const unsigned int n_q = quadrature->size();
    const unsigned int bdn_q = surf_quadrature->size();

    FEValues<dim> fe_values(*fe, *quadrature,
        update_values | update_gradients | update_quadrature_points |
        update_JxW_values);
    FEFaceValues<dim> fe_face_values(*fe, *surf_quadrature,
        update_values | update_gradients | update_JxW_values |
        update_quadrature_points | update_normal_vectors);

    // Local contribution matrix to the jacobian
    FullMatrix<double> cell_j_matrix(dofs_per_cell, dofs_per_cell);
    // Local contribution matrix to the rhs of the newton raphson iteration step
    Vector<double>     cell_nr_rhs(dofs_per_cell);

    // We will use this vector to store the global indices of the DoFs of the
    // current element within the loop.
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    std::vector<Tensor<2, dim>> shape_grad_ref(dofs_per_cell);

    std::vector<Tensor< 2, dim>> F_q(fe_values.n_quadrature_points);
    std::vector<Tensor< 2, dim>> F_q_surf(fe_face_values.n_quadrature_points);
    std::vector<std::vector<Tensor<1, dim>>> orth_u_q(fe_values.n_quadrature_points,
        std::vector<Tensor< 1, dim>>(3)
    );
    std::vector<Tensor< 1, dim>> val_u_q_surf(fe_face_values.n_quadrature_points);
    std::vector<Tensor< 2, dim>> cof_f_q_surf(fe_face_values.n_quadrature_points);

    const FEValuesExtractors::Vector displacement(0);

    std::map<types::global_dof_index, double> boundary_values;
    {
        Functions::ZeroFunction<dim> bc_function(dim);
        std::map<types::boundary_id, const Function<dim>*> boundary_functions;
        boundary_functions[PDE_DIRICHLET] = &bc_function;

        VectorTools::interpolate_boundary_values(dof_handler,
            boundary_functions,
            boundary_values);
    }

    // alignas(cache_line_size()) pass_cache_data_t intermediate {};

    // Zero out all the matrices being filled
    if (build_jacobian)
        jacobian = 0.0;
    nr_rhs_f = 0.0;

    // int prog_i = 0; TO BE USED WHEN ENABLING PROGRESS BARS!
    // const auto n = this->mesh.n_active_cells();

    alignas(cache_line_size()) pass_cache_data_t intermediate {};

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;
        // Print a progress bar every 1% of the dofs handleld
        // prog_bar_c(prog_i, n, RED_COLOR); ++prog_i;

        // Note: avoid evaluating the surface values here as most tetrahedrons
        // are not part of the surface, evaluate only inside the ifs.
        fe_values.reinit(cell);

        cell_j_matrix = 0.0;
        cell_nr_rhs = 0.0;

        fe_values[displacement].get_function_gradients(
            solution,     // The global solution vector
            F_q      // Output: The calculated gradients at all q-points
        );

        // Compute F_q
        for (std::size_t i = 0; i < F_q.size(); ++i) {
            F_q[i][0][0] += 1; F_q[i][1][1] += 1; F_q[i][2][2] += 1;
        }

        const std::vector<Point<dim>>& pts = fe_values.get_quadrature_points();

        compute_basis_at_quadrature(
            pts,
            orth_u_q, false
        );


        for (unsigned int q = 0; q < n_q; ++q)
        {
            const auto& F = F_q[q];        // Simple reference for readibility of math
            // Note: the basis if (f, s, n)
            const auto& f0 = orth_u_q[q][0];    // Reference for the fibers 
            const auto& s0 = orth_u_q[q][1];    // Reference for the sheet 

            intermediate.f0 = f0; intermediate.s0 = s0;
            // Here we assemble the local contribution for current cell and
            // current quadrature point, filling the local matrix and vector.
            const auto C = transpose(F) * F;

            compute_and_cache(i4f, scalar_product(f0, C* f0), intermediate);
            compute_and_cache(i4s, scalar_product(s0, C* s0), intermediate);
            compute_and_cache(i8sf, scalar_product(f0, C* s0), intermediate);

            compute_and_cache(macs, macaulay(i4s - 1), intermediate);
            compute_and_cache(macf, macaulay(i4f - 1), intermediate);

            compute_and_cache(exp_mac_s_sq, std::exp(bs* macs*macs), intermediate);
            compute_and_cache(exp_mac_f_sq, std::exp(bf* macf*macf), intermediate);
            compute_and_cache(exp_bi8sf_sq, std::exp(bsf* i8sf* i8sf), intermediate);

            compute_and_cache(f0f0t, outer_product(f0, f0), intermediate);
            compute_and_cache(s0s0t, outer_product(s0, s0), intermediate);

            compute_and_cache(J, determinant(F), intermediate);
            // We cache the invert explicitly (inspection of tensor.h lines 2850 give ca. 50 
            // memory accesses per inversion of matrix vs. a single cached access
            compute_and_cache(Finv, invert(F), intermediate);
            compute_and_cache(Fmt, transpose(Finv), intermediate);
            // Note: avoid caching J^-2/3, the pade approximation is fast enough and avoids
            // extra memory reads. And no need to cache I1 either since its only useful to 
            // compute exp_bi1m3
            compute_and_cache(I1, pow_m2t(J)* trace(C), intermediate);
            compute_and_cache(exp_bi1m3, std::exp(b* (I1 - 3)), intermediate);

            compute_deP_deF_at_q(F, intermediate);
            compute_P_at_q(F, intermediate);

            for (const unsigned int i : fe_values.dof_indices()) {
                const Tensor<2, dim> grad_i = fe_values[displacement].gradient(i, q);

                if (build_jacobian) {

                    for (const unsigned int j : fe_values.dof_indices()) {
                        const Tensor<2, dim> grad_j = fe_values[displacement].gradient(j, q);
                        cell_j_matrix(i, j) += fe_values.JxW(q) * scalar_product(
                            grad_i, tensor_product(deP_deF_at_q, grad_j)
                        );

                    }
                }

                cell_nr_rhs(i) += fe_values.JxW(q) * scalar_product(P_at_q, grad_i);
                
            }
        }

        // NOTE: the computation of the boundary conditions does not
        // depend on the physical problem itself, just on the stress tensor J
        // and its cofactor.
        for (unsigned int face_no = 0; face_no < cell->reference_cell().n_faces(); ++face_no) {
            // Check if this face is a boundary
            const auto id = cell->face(face_no)->boundary_id();
            if (cell->face(face_no)->at_boundary() && 
                (is_neumann(id) || is_robin(id))) {

                fe_face_values.reinit(cell, face_no);

                if (is_robin(id)) {
                    
                    val_u_q_surf.clear();
                    fe_face_values[displacement].get_function_values(
                        solution,     // The global solution vector
                        val_u_q_surf      // Output: The calculated function at all q-points
                    );

                    for (unsigned int q = 0; q < bdn_q; ++q)
                    {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                            const auto phi_i = fe_face_values[displacement].value(i, q);

                            if (build_jacobian)
                                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                                    const auto phi_j = fe_face_values[displacement].value(j, q);

                                    cell_j_matrix(i, j) += fe_face_values.JxW(q) * alfa * scalar_product(phi_i, phi_j);
                                }

                            // Rhs contribution

                            double up = alfa * scalar_product(
                                phi_i, val_u_q_surf[q]
                            ) * fe_face_values.JxW(q);
                            cell_nr_rhs(i) += up;
                        }
                    }

                }
                else if (is_neumann(id)) {

                    cof_f_q_surf.clear();
                    fe_face_values[displacement].get_function_gradients(
                        solution,
                        F_q_surf
                    );

                    for (std::size_t i = 0; i < F_q_surf.size(); ++i) {
                        // Compute F = I + grad u
                        auto& t = F_q_surf[i];
                        t[0][0] += 1; t[1][1] += 1; t[2][2] += 1;
                        // Compute F^-t
                        const auto& f_mt = transpose(invert(t));
                        cof_f_q_surf.emplace_back(f_mt);
                    }

                    // After the loop, the grad_u_q_surf vector was modified so that each
                    // element became f_q = I + grad u_q
                    auto& f_q_surf = F_q_surf;

                    // Note: we need to compute the cofactor of j = grad u at each point of 
                    // the boundary quadrature due to Robin's condition of the domain (also Neumann)
                    for (unsigned int q = 0; q < bdn_q; ++q)
                    {
                        const auto normal_q = fe_face_values.normal_vector(q);
                        const auto d = determinant(f_q_surf[q]);
                        const auto cofactor = cof_f_q_surf[q] * d;

                        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                            const auto phi_i = fe_face_values[displacement].value(i, q);
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                                const auto grad_j = fe_face_values[displacement].gradient(j, q);

                                if (build_jacobian)
                                    cell_j_matrix(i, j) += fe_face_values.JxW(q) *
                                    p_v * scalar_product(phi_i,  
                                        (
                                            scalar_product(cof_f_q_surf[q], grad_j) * dealii::Physics::Elasticity::StandardTensors<dim>::I +
                                            -cof_f_q_surf[q] * transpose(grad_j)
                                        ) * cofactor * normal_q
                                    );

                            }


                            // RHS contribution

                            double up = p_v * fe_face_values.JxW(q) * scalar_product(
                                phi_i, cofactor * normal_q
                            );
                            cell_nr_rhs(i) += up;
                        }


                    }
                }
            }
        }
        // Now we must compute the right hand side of this, namely the F_i values we need
        // to put to zero with the newton raphson iteration e.g. compute the contribution for the 
        // i-th basis function



        cell->get_dof_indices(dof_indices);

        constraints.distribute_local_to_global(
             cell_j_matrix, cell_nr_rhs, dof_indices, jacobian, nr_rhs_f);
        // Uncomment this code to use the serial-implementation for the code (dont)
        // constraints.distribute_local_to_global(cell_j_matrix, cell_nr_rhs, dof_indices, jacobian, nr_rhs_f);
        // jacobian.add(dof_indices, cell_j_matrix);
        // nr_rhs_f.add(dof_indices, cell_nr_rhs);

    }
    jacobian.compress(VectorOperation::add);
    nr_rhs_f.compress(VectorOperation::add);

    //  Uncomment this code section to apply boundary values in here and not in solve().
    // MatrixTools::apply_boundary_values(
    //    boundary_values, jacobian, step_owned, nr_rhs_f, true);

    // Util macro... ends the progress bar, cannot be used for parallel program
    // end_prog_bar();

}

double OrthotropicSolver::compute_internal_volume() {
    // Just use volume integration
    return 0.0;
}

double OrthotropicSolver::compute_external_volume() {
    // TODO: To be done when simulating the pressure cycle, use divergence theorem
    // on the endocardium
    return 0.0;
}


void
OrthotropicSolver::compute_basis_at_quadrature(
    const std::vector<Point<dim>>& p,
    std::vector<std::vector<dealii::Tensor<1, dim>>>& orth_sys,
    bool compute_n
) {
    // Simply loop over all entries.
    for (unsigned int i = 0; i < p.size(); ++i)
        orthothropic_base_at(p[i], orth_sys[i], compute_n);
    return;
}

void
OrthotropicSolver::orthothropic_base_at(
    const dealii::Point<dim>& p, std::vector<dealii::Tensor<1, dim>>& basis, bool compute_n
) {
    const double x = p[0], y = p[1], z = p[2];

    const double csquared = MESH_ELLIPSOID_Z_DEFORMATION * MESH_ELLIPSOID_Z_DEFORMATION;
    const double cfourthp = csquared * csquared;

    const double r = std::sqrt(x * x + y * y + z * z / cfourthp);
    const double rr = std::sqrt(x * x + y * y);
    // Let s be the radial vector aligned with the collagen sheet. 
    auto& s = basis[1];
    s[0] = x / r; s[1] = y / r; s[2] = (z / csquared) / r;
    // FLAG[GEO] 
    double corrected_radius = std::sqrt(x * x + y * y + z * z / csquared);
    double r_over_width = (corrected_radius - MESH_ELLIPSOID_SMALL_RADIUS) / (MESH_ELLIPSOID_LARGE_RADIUS - MESH_ELLIPSOID_SMALL_RADIUS);
    r_over_width = (r_over_width < 0) ? 0 : (r_over_width > 1) ? 1 : r_over_width;

    double rads = (r_over_width - 0.5) * (r_over_width - 0.5) * (r_over_width - 0.5) * 8 * 1.03;
    rads = (r_over_width - 0.5) * 2 * 1.03;

    auto& f = basis[0];
    f[0] = -y / rr; f[1] = x / rr; f[2] = 0;

    auto& n = basis[2];
    n = cross_product_3d(f, s);

    // Rotate F in the tangent plane by the transmurally-variant degree
    f = f * std::cos(rads) + n * std::sin(rads);
    if (compute_n)
        n = cross_product_3d(s, f);

}