#include <cmath>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/mapping_q1_eulerian.h>

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

    /*
    UtilsMesh::visualize_wall_depth<3>(mesh, "depthVisuals.vtu");
    UtilsMesh::visualize_grain_fibers(
        [this](const dealii::Point<3, double>& p, std::vector<dealii::Tensor<1, 3>>& b) {
            orthothropic_base_at(p, b, true);
        },
        mesh, fe, quadrature, "fibersVisuals.vtu");
        */
}

constexpr const auto dim = OrthotropicSolver::dim;

void
OrthotropicSolver::step_pressure(unsigned int n_steps, double pressure_) {

    if (pressure_)
        p_v = pressure_;
    // Remove any activbe contraction for the initial pressure step. 
    double save_beta = 0, save_s_n = 0;
    double final_pressure = p_v;

    p_v = 0;
    std::swap(beta, save_beta);
    std::swap(Sn, save_s_n);

    pde_out_c_par(pcout, "Stepping up the pressure: ", RED_COLOR);



    ReductionControl solver_control(
        key("Max iterations", 5000),
        key("Tolerance", 1.0e-9)
    );
    // We use GMRES as the matrix is in general non symmetric. 
    SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

    constexpr const auto NEWTON_ITER = 10;

    for (int step = 0; step < n_steps + 1; step++) {
        // TODO: (just a little thing) add a break when a given threshold residual
        // is achieved at each load_stepping step!
        for (unsigned int newton_iter = 0; newton_iter < NEWTON_ITER; ++newton_iter) {

            const double alpha_k = 0.45 + 0.20 * (newton_iter) / NEWTON_ITER;

            pde_out_c_par(pcout, "Entering step number: " << newton_iter, YEL_COLOR);

            this->build_system();

            TrilinosWrappers::PreconditionAMG precondition_amg;
            // precondition_amg.initialize(jacobian);

            TrilinosWrappers::PreconditionILU::AdditionalData precondition_data;
            precondition_data.ilu_fill = 2;           // Fill-in level (0=diagonal, 2=good)

            TrilinosWrappers::PreconditionILU precondition_ilu;
            precondition_ilu.initialize(jacobian, precondition_data);
            try {
                solver.solve(jacobian, step_owned, nr_rhs_f, precondition_ilu);
            }
            catch (const SolverControl::NoConvergence& e) {
                pde_out_c_par(pcout, "FAILED to solve the GMRES iteration at step " << newton_iter, RED_COLOR);
                break;
            }
            pde_out_c_par(pcout, "Complete after " << solver_control.last_step() << " GMRES iterations", RED_COLOR);
            solution_owned.add(-alpha_k, step_owned);
            constraints.distribute(solution_owned);

            solution = solution_owned;

            for (int NR = 0; NR < 0; ++NR) {
                this->build_system(false);
                solver.solve(jacobian, step_owned, nr_rhs_f, precondition_amg);
                solution_owned.add(-alpha_k / 3, step_owned);
                constraints.distribute(solution_owned);
                solution = solution_owned;
            }

        }
        // Incomplete placeholder for pressure load stepping: 
        // just sub with max_press - init_press / It_amt
        if (step < n_steps)
            p_v += final_pressure / n_steps;
    }

    pde_out_c_par(pcout, "Completed the newton iteration at pressure " << p_v << ", saving the result", GRN_COLOR);
    {
        DataOut<dim> data_out;

        data_out.add_data_vector(dof_handler, solution, "solution");

        std::vector<unsigned int> partition_int(mesh.n_active_cells());
        GridTools::get_subdomain_association(mesh, partition_int);
        const Vector<double> partitioning(partition_int.begin(), partition_int.end());
        data_out.add_data_vector(partitioning, "partitioning");

        data_out.build_patches();

        data_out.write_vtu_with_pvtu_record("./",
            "PressureStep.vtk", 0,
            MPI_COMM_WORLD);

        pde_out_c_par(pcout, "Result saved into file PressureStep.vtk", GRN_COLOR);
    }

    // Restore the active formulation. 
    std::swap(beta, save_beta);
    std::swap(Sn, save_s_n);
}

void
OrthotropicSolver::step_active(unsigned int n_steps, double S_n_, double beta_) {

    if (S_n_)
        Sn = S_n_;
    if (beta_)
        beta = beta_;
    double final_Sn = Sn, final_beta = beta;

    Sn = beta = 0.0;
    pde_out_c_par(pcout, "Stepping up the active components: ", RED_COLOR);
    ReductionControl solver_control(
        key("Max iterations", 5000),
        key("Tolerance", 1.0e-9)
    );
    // We use GMRES as the matrix is in general non symmetric. 
    SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

    constexpr const auto NEWTON_ITER = 10;

    for (int step = 0; step < n_steps; step++) {
        // TODO: (just a little thing) add a break when a given threshold residual
        // is achieved at each load_stepping step!
        for (unsigned int newton_iter = 0; newton_iter < NEWTON_ITER; ++newton_iter) {

            const double alpha_k = 0.35 + 0.20 * (newton_iter) / NEWTON_ITER;

            pde_out_c_par(pcout, "Entering step number: " << newton_iter, YEL_COLOR);

            this->build_system();



            TrilinosWrappers::PreconditionILU::AdditionalData precondition_data;
            precondition_data.ilu_fill = 2;           // Fill-in level (0=diagonal, 2=good)

            TrilinosWrappers::PreconditionILU precondition_ilu;
            precondition_ilu.initialize(jacobian, precondition_data);
            try {
                solver.solve(jacobian, step_owned, nr_rhs_f, precondition_ilu);
            }
            catch (const SolverControl::NoConvergence& e) {
                pde_out_c_par(pcout, "FAILED to solve the GMRES iteration at step " << newton_iter, RED_COLOR);
                break;
            }

            solution_owned.add(-alpha_k, step_owned);
            constraints.distribute(solution_owned);

            solution = solution_owned;

            for (int NR = 0; NR < 0; ++NR) {
                this->build_system(false);
                solver.solve(jacobian, step_owned, nr_rhs_f, precondition_ilu);
                solution_owned.add(-alpha_k / 3, step_owned);
                constraints.distribute(solution_owned);
                solution = solution_owned;
            }

        }
        // Incomplete placeholder for pressure load stepping: 
        // just sub with max_press - init_press / It_amt
        Sn += final_Sn / n_steps;
        beta += final_beta / n_steps;
    }

    pde_out_c_par(pcout, "Completed the newton iteration, saving the result", GRN_COLOR);
    {
        DataOut<dim> data_out;

        data_out.add_data_vector(dof_handler, solution, "solution");

        std::vector<unsigned int> partition_int(mesh.n_active_cells());
        GridTools::get_subdomain_association(mesh, partition_int);
        const Vector<double> partitioning(partition_int.begin(), partition_int.end());
        data_out.add_data_vector(partitioning, "partitioning");

        data_out.build_patches();

        data_out.write_vtu_with_pvtu_record("./",
            "ActiveStep.vtk", 0,
            MPI_COMM_WORLD);

        pde_out_c_par(pcout, "Result saved into file ActiveStep.vtk", GRN_COLOR);
    }

}



void 
OrthotropicSolver::contraction() {

    constexpr const auto tol = 100;
    constexpr const auto delta_t = 0.05;

    constexpr const auto final_time = 1.0;
    constexpr const auto t0 = 0.3;
    const auto Sn_save = Sn;

    double t = 0.0;

    const auto Sn_func = [Sn_save, t0, final_time](double t) {
        return Sn_save * std::sin(M_PI * (t - t0) / (final_time - t0)) * std::sin(M_PI * (t - t0) / (final_time - t0));
        };

    // Assume the ventricle is initially filled. 
    Sn = 0.0;

    solve("time_-1" + std::to_string(t));
    const auto V_ed = compute_external_volume();
    double pressure = p_v;

    std::ofstream outfile;

    if (mpi_rank == 0) {
        outfile.open("output.txt", std::ios::app);  // append mode
        if (!outfile.is_open()) {
            std::cerr << "Error opening file!" << std::endl;
        }
    }

    t = t0;
    while (t < final_time) {

        if (p_v < 2.1) {

            Sn = Sn_func(t);
            solve("time_" + std::to_string(t), 1e-1);

            pde_out_c_par(pcout, "ACTIVE COMPONENT " << Sn << ", " << beta, BLU_COLOR);

            auto vol = compute_external_volume();
            pde_out_c_par(pcout, "AT TIME " << t << " PRESSURE " << pressure << " VOLUME" << vol, BLU_COLOR);

            while (std::abs(vol - V_ed) > tol) {
                const auto C_km1 = vol / pressure;

                pde_out_c_par(pcout, "VOLUME DIFFERENCE " << (vol - V_ed), RED_COLOR);

                pressure = pressure - (vol - V_ed) / C_km1;
                pde_out_c_par(pcout, "NEW PRESSURE " << (pressure), RED_COLOR);

                p_v = pressure;
                solve("time_" + std::to_string(t), 1e-1);

                vol = compute_external_volume();
            }
            if (mpi_rank == 0) {
                outfile << vol << " " << p_v << " " << t << std::endl;
                outfile.flush();  // Force write to disk
            }
        }
        else {
            // Ejection of the ventricle. 
            constexpr const auto Cwk = 0.00165; // mL / kPa
            constexpr const auto Rwk = 80   ; // ms / mL
            // Pressure in kPa. 
            // compute_external_volume() gives volume in millimiter cubed. 
            // 1000 mm = 1 mL
            Sn = Sn_func(t);
            solve("time_" + std::to_string(t));

            // Before time loop: keep track of previous state
            double P_old = p_v;
            double V_old = compute_external_volume(); // mm^3
            // Inside ejection branch, for each time t:
            
            double P = P_old;

            const double dt = delta_t;   // seconds
            const double dt_ms = dt * 1000.0;

            const double V_old_ml = V_old / 1000.0;

            for (unsigned int it = 0; it < 20; ++it)
            {
                // --- evaluate F(P) ---

                p_v = P;
                solve("time_" + std::to_string(t), 1e-4);
                pde_out_c_par(pcout, "WindKESSEL ON" << p_v, BLU_COLOR);

                const double V_now = compute_external_volume();
                const double V_now_ml = V_now / 1000.0;

                const double F =
                    Cwk * (P - P_old)
                    + (dt_ms / Rwk) * P
                    + (V_now_ml - V_old_ml);

                if (std::abs(F) < 1e-6)
                    break;

                // --- numerical derivative ---

                const double eps = 1e-3;

                p_v = P + eps;
                solve("time_" + std::to_string(t), 1e-4);

                const double V_eps = compute_external_volume() / 1000.0;

                const double F_eps =
                    Cwk * (P + eps - P_old)
                    + (dt_ms / Rwk) * (P + eps)
                    + (V_eps - V_old_ml);

                const double dF = (F_eps - F) / eps;

                // Newton update
                P -= F / dF;
            }

            // After convergence, accept P, update state
            p_v = P;
            V_old = compute_external_volume();
            P_old = P;
            pde_out_c_par(pcout, "PRESSURE AND VOLUME: " << p_v << ", " << V_old, BLU_COLOR);

            // Logging
            if (mpi_rank == 0) {
                outfile << V_old << " " << p_v << " " << t << std::endl;
                outfile.flush();
            }
        }
        t += delta_t;
    }

}

void
OrthotropicSolver::solve(const std::string& output_file_name, double tol ) {

    const unsigned int dofs_per_cell = fe->dofs_per_cell;

    pde_out_c_par(pcout, "Solving the non linear mechanics: ", RED_COLOR);
    pde_out_c_par(pcout, "DOFS per Cell " << dofs_per_cell, RED_COLOR);

    constexpr const auto MAX_ITER_AMT = 70;
    //     solution = 0.0;

    ReductionControl solver_control(
        key("Max iterations", 5000),
        key("Tolerance", 1.0e-9)
    );
    // We use GMRES as the matrix is in general non symmetric. 
    SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

        // TODO: (just a little thing) add a break when a given threshold residual
        // is achieved at each load_stepping step!
    double res = 1.0;
    unsigned int newton_iter = 0;
    while (res > tol && newton_iter < MAX_ITER_AMT) {
        //for (unsigned int newton_iter = 0; newton_iter < MAX_ITER_AMT; ++newton_iter) {

        const double alpha_k = 0.15 + 0.20 * (newton_iter) / MAX_ITER_AMT;

        // pde_out_c_par(pcout, "Assembling the linear system", RED_COLOR);

        this->build_system();

        // pde_out_c_par(pcout, "Solving the linear system:", RED_COLOR);
        res = nr_rhs_f.l2_norm();
        pde_out_c_par(pcout, newton_iter << ") L2 Norm of residual: " << res, RED_COLOR);
        // pde_out_c_par(pcout, "L2 Norm of Jacobian: " << jacobian.l1_norm(), RED_COLOR);

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
        // pde_out_c_par(pcout, "Complete after " << solver_control.last_step() << " GMRES iterations", RED_COLOR);
        // pde_out_c_par(pcout, "L2 Norm of the step: " << step_owned.l2_norm(), RED_COLOR);

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
        // pde_out_c_par(pcout, "Beginning Quasi Newton rounds...", RED_COLOR);
        for (int NR = 0; NR < 0; ++NR) {
            this->build_system(false);
            solver.solve(jacobian, step_owned, nr_rhs_f, precondition_amg);
            solution_owned.add(-alpha_k/3, step_owned);
            constraints.distribute(solution_owned);
            solution = solution_owned;
        }
        // At each step, ensure the solution has the right constraints. Use the
        // object oriented version to just limit the z component of the displacement
        // constraints.distribute(solution);

    }
    // Incomplete placeholder for pressure load stepping: 
    // just sub with max_press - init_press / It_amt
    // p_v += 0.17;
    

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
    return Sn * (1 + beta * (std::sqrt(i4) - 1.0)) / (i4);
}

double
OrthotropicSolver::active_phi_prime(const double i4) {
    return (Sn / (i4)) * ((-(1 + beta * (sqrt(i4)-1.0)) / i4) + b/(2*sqrt(i4)));
    // return (-3 / 2.0) * Sn * (1 + beta * (std::sqrt(i4) - 1)) / (i452)+
    //     Sn * beta / (2 * i4*i4);
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

            if (D.i4f > 1) {
                // Active term
                r += active_phi_prime(D.i4f) * Ff0f0t[i][j] * Ff0f0t + active_phi(D.i4f) * f * D.f0f0t;
            }
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

void OrthotropicSolver::compute_radius()
{
    solution.update_ghost_values();

    MappingQEulerian<dim, TrilinosWrappers::MPI::Vector> mapping(
        fe->degree,
        dof_handler,
        solution);

    FEFaceValues<dim> fe_face(
        mapping,
        *fe,
        *surf_quadrature,
        update_quadrature_points);

    double local_r_epi = 0.0;
    double local_r_endo = 1e5;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
            if (!cell->face(f)->at_boundary())
                continue;

            const auto bid = cell->face(f)->boundary_id();

            if (!is_dirichlet(bid))
                continue;

            fe_face.reinit(cell, f);

            for (unsigned int q = 0; q < fe_face.n_quadrature_points; ++q)
            {
                const Point<dim>& x = fe_face.quadrature_point(q);

                const double r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
                pcout << "RAD: " << r << " x " << x[0] << " y " << x[1] << " z " << x[2] << std::endl;
                local_r_endo = std::min(local_r_endo, r);
                local_r_epi = std::max(local_r_epi, r);

            }
        }
    }

    pde_out_c_par(pcout,
        "Local Epi diameter = " << local_r_epi <<
        "  Endo diameter = " << local_r_endo,
        YEL_COLOR);


    const double r_endo = Utilities::MPI::min(local_r_endo, MPI_COMM_WORLD);
    const double r_epi = Utilities::MPI::max(local_r_epi, MPI_COMM_WORLD);

    double D_epi = 2.0 * r_epi;
    double D_endo = 2.0 * r_endo;

    pde_out_c_par(pcout,
        "Epi diameter = " << D_epi <<
        "  Endo diameter = " << D_endo,
        YEL_COLOR);

    pde_out_c_par(pcout,
        "Wall thickness = " << D_epi-D_endo,
        YEL_COLOR);
}


void OrthotropicSolver::compute_height()
{
    solution.update_ghost_values();

    MappingQEulerian<dim, TrilinosWrappers::MPI::Vector> mapping(
        fe->degree,
        dof_handler,
        solution);

    FEFaceValues<dim> fe_face(
        mapping,
        *fe,
        *surf_quadrature,
        update_quadrature_points);

    double local_zmax = -1e30;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
            if (!cell->face(f)->at_boundary())
                continue;

            const auto bid = cell->face(f)->boundary_id();

            if (!is_robin(bid))
                continue;

            fe_face.reinit(cell, f);

            for (unsigned int q = 0; q < fe_face.n_quadrature_points; ++q)
            {
                const Point<dim>& x = fe_face.quadrature_point(q);

                local_zmax = std::max(local_zmax, x[2]);
            }
        }
    }
    const double zmax = Utilities::MPI::max(local_zmax, MPI_COMM_WORLD);

    double height = zmax;

    pde_out_c_par(pcout, "Computed the height to be: " << height,
        YEL_COLOR);
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

                            double up = +alfa * scalar_product(
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

// Just use volume integration
double OrthotropicSolver::compute_internal_volume() {

    pde_out_c_par(pcout, "Computing the internal volume...", YEL_COLOR);
    // Number of quadrature points for each element.
    const unsigned int n_q = quadrature->size();


    TrilinosWrappers::MPI::Vector ghosted_solution;
    const IndexSet& locally_owned_dofs = dof_handler.locally_owned_dofs();
    const IndexSet& locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);

    ghosted_solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

    ghosted_solution = solution;
    ghosted_solution.update_ghost_values();

    MappingQEulerian<dim, TrilinosWrappers::MPI::Vector> mapping(
        fe->degree,
        dof_handler,
        ghosted_solution);

    FEValues<dim> fe_values(mapping, *fe, *quadrature,
        update_values | update_quadrature_points | update_JxW_values);

    double local_volume_acc = 0;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;
        // Print a progress bar every 1% of the dofs handleld
        // prog_bar_c(prog_i, n, RED_COLOR); ++prog_i;

        // Note: avoid evaluating the surface values here as most tetrahedrons
        // are not part of the surface, evaluate only inside the ifs.
        fe_values.reinit(cell);

        for (unsigned int q = 0; q < n_q; ++q)
        {            
            local_volume_acc += fe_values.JxW(q);
        }
    }

    // MPI sum over all processors
    const double global_volume =
        Utilities::MPI::sum(local_volume_acc,
            MPI_COMM_WORLD);

    pde_out_c_par(pcout, "Computed the internal volume to be: " << global_volume, YEL_COLOR);

    return global_volume;
}

double OrthotropicSolver::compute_external_volume() {

    pde_out_c_par(pcout, "Computing the external volume...", YEL_COLOR);
    double local_volume = 0;
    FEFaceValues<dim> fe_face_values(*fe, *surf_quadrature,
        update_values | update_gradients | update_normal_vectors | update_quadrature_points | update_JxW_values);


    std::vector<Tensor<1, dim>> u_values(surf_quadrature->size());
    std::vector<Tensor<2, dim>> grad_u(surf_quadrature->size());

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned())
            continue;

        for (unsigned int f = 0; f < cell->n_faces(); ++f) {
            if (cell->face(f)->at_boundary() && is_neumann(cell->face(f)->boundary_id())) {
                fe_face_values.reinit(cell, f);

                fe_face_values[FEValuesExtractors::Vector(0)].get_function_values(solution, u_values);
                fe_face_values[FEValuesExtractors::Vector(0)].get_function_gradients(solution, grad_u);

                for (unsigned int q = 0; q < surf_quadrature->size(); ++q) {
                    const Point<dim> X = fe_face_values.quadrature_point(q);
                    const Tensor<1, dim> x_current = X + u_values[q];

                    Tensor<2, dim> F = Physics::Elasticity::StandardTensors<dim>::I + grad_u[q];
                    const double   J = determinant(F);
                    Tensor<2, dim> F_inv_T = transpose(invert(F));

                    const Tensor<1, dim> N = fe_face_values.normal_vector(q);
                    // Deformed area vector via Nanson: n da = J F^{-T} N dA
                    const Tensor<1, dim> n_da = J * F_inv_T * N;

                    // We need the "current" normal n_cur * da_cur. BEFORE
                    // WE WERE USING THE UNDEFORMED NORMAL...
                    // Using Nanson's formula: n da = J * F^-T * N * dA
                    local_volume += (x_current / 3.0 * -n_da) * fe_face_values.JxW(q);
                }
            }
        }

    }

    // MPI sum over all processors
    const double global_volume =
        Utilities::MPI::sum(local_volume,
            MPI_COMM_WORLD);

    pde_out_c_par(pcout, "Computed the external volume to be: " << global_volume, YEL_COLOR);
    return global_volume;

}

void OrthotropicSolver::compute_strains(const std::string& save_path) {
    
    const FEValuesExtractors::Vector displacement(0);
    
    FEValues<dim> fe_values(*fe, *quadrature,
        update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int n_q = quadrature->size();
    std::vector<Tensor<2, dim>> grad_u(n_q);

    // Storage for projected strains — one value per cell (averaged over quadrature points)
    // You can also store per-quadrature-point if you prefer
    Vector<double> E_cc_vec(mesh.n_active_cells());
    Vector<double> E_ll_vec(mesh.n_active_cells());
    Vector<double> E_rr_vec(mesh.n_active_cells());
    Vector<double> E_cl_vec(mesh.n_active_cells());
    Vector<double> E_cr_vec(mesh.n_active_cells());
    Vector<double> E_lr_vec(mesh.n_active_cells());

    for (const auto& cell : dof_handler.active_cell_iterators()) {

        if (!cell->is_locally_owned()) continue;

        fe_values.reinit(cell);
        fe_values[displacement].get_function_gradients(solution, grad_u);

        double Ecc = 0.0, Ell = 0.0, Err = 0.0;
        double Ecl = 0.0, Ecr = 0.0, Elr = 0.0;
        double total_weight = 0;

        for (unsigned int q = 0; q < n_q; ++q) {
            const Point<dim> X = fe_values.quadrature_point(q);
            const double w = fe_values.JxW(q);

            // Deformation gradient and right Cauchy-Green tensor
            Tensor<2, dim> F = Physics::Elasticity::StandardTensors<dim>::I + grad_u[q];
            Tensor<2, dim> C = transpose(F) * F;
            Tensor<2, dim> E = 0.5 * (C - Physics::Elasticity::StandardTensors<dim>::I);

            // Build cardiac coordinate system at this point
            // Circumferential: tangent to ellipsoid in XY plane, counterclockwise
            // Longitudinal: pointing from apex to base (z direction)  
            // Radial: pointing outward from the long axis
            Tensor<1, dim> e_r, e_c, e_l;

            const double x = X[0], y = X[1], z = X[2];

            const double csquared = MESH_ELLIPSOID_Z_DEFORMATION * MESH_ELLIPSOID_Z_DEFORMATION;
            const double cfourthp = csquared * csquared;

            const double r = std::sqrt(x * x + y * y + z * z / cfourthp);
            const double rr = std::sqrt(x * x + y * y);
            // Let s be the radial vector aligned with the collagen sheet. 
            e_r[0] = x / r; e_r[1] = y / r; e_r[2] = (z / csquared) / r;
            e_c[0] = -y / rr; e_c[1] = x / rr; e_c[2] = 0;
            e_l = cross_product_3d(e_c, e_r);

            // Project E onto cardiac coordinate system
            // E_ij = e_i · E · e_j
            Ecc += (e_c * (E * e_c)) * w;
            Ell += (e_l * (E * e_l)) * w;
            Err += (e_r * (E * e_r)) * w;
            Ecl += (e_c * (E * e_l)) * w;
            Ecr += (e_c * (E * e_r)) * w;
            Elr += (e_l * (E * e_r)) * w;

            total_weight += w;
        }

        // Volume-average over the cell
        const unsigned int idx = cell->active_cell_index();

        E_cc_vec[idx] = Ecc / total_weight;
        E_ll_vec[idx] = Ell / total_weight;
        E_rr_vec[idx] = Err / total_weight;
        E_cl_vec[idx] = Ecl / total_weight;
        E_cr_vec[idx] = Ecr / total_weight;
        E_lr_vec[idx] = Elr / total_weight;
    }

    // Add to DataOut for visualization
    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution, "solution");
    data_out.add_data_vector(E_cc_vec, "E_cc");
    data_out.add_data_vector(E_ll_vec, "E_ll");
    data_out.add_data_vector(E_rr_vec, "E_rr");
    data_out.add_data_vector(E_cl_vec, "E_cl");
    data_out.add_data_vector(E_cr_vec, "E_cr");
    data_out.add_data_vector(E_lr_vec, "E_lr");
    data_out.build_patches();
    // write as usual...

    data_out.write_vtu_with_pvtu_record("./",
        std::to_string(alfa) + "_" + save_path, 0,
        MPI_COMM_WORLD);

    pde_out_c_par(pcout, "Result of strain computation saved into file " << save_path, GRN_COLOR);

}

// Pseudocode for calculating rotation at a specific Z-coordinate
double 
OrthotropicSolver::calculate_slice_rotation(double target_z, double tolerance) {

    double total_phi = 0;
    unsigned int count = 0;

    FEValues<dim> fe_values(*fe, *quadrature,
        update_values | update_normal_vectors | update_quadrature_points | update_JxW_values);
    std::vector<Tensor<1, dim>> u_values(quadrature->size());

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) 
            continue;

        fe_values.reinit(cell);

        // Get displacement values to find current position x = X + u
        fe_values[FEValuesExtractors::Vector(0)].get_function_values(solution, u_values);

        if (std::abs(cell->center()[2] - target_z) > tolerance)
            continue;

        for (unsigned int q = 0; q < surf_quadrature->size(); ++q) {

            const Point<dim> X = fe_values.quadrature_point(q);
            const Tensor<1, dim> x_current = X + u_values[q];

            // Reference and deformed angles in the XY plane
            double theta_ref = std::atan2(X[1], X[0]);
            double theta_def = std::atan2(x_current[1], x_current[0]);

            double d_phi = theta_def - theta_ref;

            // Handle atan2 wrapping (-pi to pi)
            if (d_phi > M_PI) d_phi -= 2 * M_PI;
            if (d_phi < -M_PI) d_phi += 2 * M_PI;

            total_phi += d_phi;
            count++;
           
        }
    }
    // Synchronize across MPI processors
    const double global_angle = Utilities::MPI::sum(total_phi, MPI_COMM_WORLD) /
        Utilities::MPI::sum(count, MPI_COMM_WORLD);
    pde_out_c_par(pcout, "(z=" << target_z << ")Computed the twist to be: " << global_angle << "(" << 180*(global_angle/M_PI) << "deg)", YEL_COLOR);

    return global_angle;
}

double
OrthotropicSolver::compute_unperturbed_volume() {
    const double volume = GridTools::volume(mesh);
    pde_out_c_par(pcout, "Computed the resting volume to be:" << volume, YEL_COLOR);
    return volume;
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

double compute_point_depth(const dealii::Point<dim>& p) {

    const double x = p[0], y = p[1], z = p[2];

    double d = 29.1;
    double z_cut = 11.9;
    double xi_endo = 0.6;
    double xi_epi = 1.02;

    double r_plus = std::sqrt(x * x + y * y + (z - z_cut - d) * (z - z_cut - d));
    double r_minus = std::sqrt(x * x + y * y + (z - z_cut + d) * (z - z_cut + d));

    double cosh_xi = (r_plus + r_minus) / (2.0 * d);
    double xi = std::acosh(cosh_xi);

    double depth = (xi - xi_endo) / (xi_epi - xi_endo);
    depth = (depth < 0) ? 0 : (depth > 1) ? 1 : depth;


    return depth;
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
    double d = 29.1;
    double z_cut = 11.9;

    double r_plus = std::sqrt(x * x + y * y + (z - z_cut - d) * (z - z_cut - d));
    double r_minus = std::sqrt(x * x + y * y + (z - z_cut + d) * (z - z_cut + d));

    double cosh_xi = (r_plus + r_minus) / (2.0 * d);
    double sinh_xi = std::sqrt(cosh_xi * cosh_xi - 1.0);

    double common = 1.0 / (2.0 * d * sinh_xi);
    double dxi_dx = common * x * (1.0 / r_plus + 1.0 / r_minus);
    double dxi_dy = common * y * (1.0 / r_plus + 1.0 / r_minus);
    double dxi_dz = common * ((z - z_cut - d) / r_plus + (z - z_cut + d) / r_minus);

    double grad_xi_norm = std::sqrt(dxi_dx * dxi_dx + dxi_dy * dxi_dy + dxi_dz * dxi_dz);
    s[0] = dxi_dx / grad_xi_norm;
    s[1] = dxi_dy / grad_xi_norm;
    s[2] = dxi_dz / grad_xi_norm;

    double r_over_width = compute_point_depth(p);

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