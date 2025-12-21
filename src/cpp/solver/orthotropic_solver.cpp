#if 0

#include <cmath>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/lac/sparse_ilu.h>
#include <fstream>
#include <iostream>
#include <functional>

#include "orthotropic_solver.hpp"
#include "mesh_geometry.hpp"
#include "boundaries.hpp"

// Uncomment this definition to get verbose output (datetime and code line)
#define PDE_OUT_VERBOSE
#define PDE_PROGRESS_BAR
#include "../utilities/visualize.hpp"
#include "../utilities/mesh_io.hpp"
#include "../utilities/math.hpp"

// Just used to enhance readibility of functions
#define key(s1, s) s

using namespace dealii;

void
SuperElasticOrthotropicSolver::setup(
    const std::string& mesh_path
) {

    {
        pde_out_c("Loading mesh " << mesh_path, RED_COLOR);
        UtilsMesh::load_mesh_into_tria(mesh_path, this->mesh);
    }
    {
        pde_out_c("Initializing the finite element space", YEL_COLOR);

        fe = std::make_unique<FESystem<dim>>(FE_SimplexP<dim>(this->r_deg) ^ dim);

        pde_out_c_i("Degree = " << fe->degree, YEL_COLOR, 1);
        pde_out_c_i("DoFs per cell = " << fe->dofs_per_cell, YEL_COLOR, 1);

        quadrature = std::make_unique<QGaussSimplex<dim>>(r_deg + 1);
        surf_quadrature = std::make_unique<QGaussSimplex<dim - 1>>(r_deg + 1);

        pde_out_c_i("Volume quadrature points per cell = " << quadrature->size(), YEL_COLOR, 1);
        pde_out_c_i("Surface quadrature points per cell = " << surf_quadrature->size(), YEL_COLOR, 1);

    }
    {
        pde_out_c("Initializing the DoF handler", BLU_COLOR);

        dof_handler.reinit(mesh);

        dof_handler.distribute_dofs(*fe);
        pde_out_c_i("Number of DoFs = " << dof_handler.n_dofs(), BLU_COLOR, 1);
    }
    {

        constraints.clear();
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

        pde_out_c("Initializing the sparsity pattern", GRN_COLOR);
        DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

        DoFTools::make_sparsity_pattern(dof_handler, dsp);

            // , constraints, true);
        pde_out_c("Copying the sparsity pattern", GRN_COLOR);
        sparsity_pattern.copy_from(dsp);
        jacobian.reinit(sparsity_pattern);

        pde_out_c_i("Initializing the solution vector", GRN_COLOR, 1);
        solution.reinit(dof_handler.n_dofs());
        step.reinit(dof_handler.n_dofs());
        nr_rhs_f.reinit(dof_handler.n_dofs());
    }
}

constexpr const auto dim = SuperElasticOrthotropicSolver::dim;


#define compute_and_cache(name, expression, inter) const auto name = expression; inter.name = name; 

void
SuperElasticOrthotropicSolver::solve() {

    const unsigned int dofs_per_cell = fe->dofs_per_cell;

    pde_out_c("Solving the non linear mechanics: ", RED_COLOR);
    pde_out_c("DOFS per Cell " << dofs_per_cell,    RED_COLOR);

    // Some utility visualization functions to ensure that everything makes 
    // physical sense
    {
        UtilsMesh::view_cartesian_coords(this->mesh, fe, quadrature, "field_new.vtu");
        UtilsMesh::visualize_grain_fibers(
            [this](const Point<dim>& p, std::vector<Tensor<1, dim>>& v) { return this->orthothropic_base_at(p, v, true); },
            this->mesh, fe, quadrature, "orth_field_new.vtu"
        );
        UtilsMesh::visualize_wall_depth(this->mesh, "depth.vtu");
    }

#define MAX_ITER_AMT 10
    solution = 0.0;

    ReductionControl solver_control(
        key("Max iterations", 1000),
        key("Tolerance", 1.0e-16),
        key("Reduce", 1.0e-6)
    );
    // We use GMRES as the matrix is in general non symmetric. 
    SolverGMRES<Vector<double>> solver(solver_control);

    for (unsigned int newton_iter = 0; newton_iter < MAX_ITER_AMT; ++newton_iter) {

        const double alpha_k = 0.3 + 0.5 * (newton_iter) / MAX_ITER_AMT;

        pde_out_c("Entering step number: " << newton_iter,  YEL_COLOR);
        pde_out_c("Assembling the linear system",           RED_COLOR);

        this->build_system();

        pde_out_c("Solving the linear system:",                         RED_COLOR);
        pde_out_c("L2 Norm of residual: " << nr_rhs_f.l2_norm(),        RED_COLOR);
        pde_out_c("L2 Norm of Jacobian: " << jacobian.l1_norm(),        RED_COLOR);

        /*  ---- Notice ----
        * When using a non object oriented constraint (e.g. not just the z component, 
        * uncomment this section of the code */
        /* MatrixTools::apply_boundary_values(
            boundary_values, jacobian, step, nr_rhs_f, false
        ); */

        SparseILU<double> precondition;
        precondition.initialize(
            jacobian, 
            SparseILU<double>::AdditionalData(0, key("Additional non-zero diags", 2))
        );
        try {
            solver.solve(jacobian, step, nr_rhs_f, precondition);
        }
        catch (const SolverControl::NoConvergence& e) {
            pde_out_c("FAILED to solve the GMRES iteration at step " << newton_iter, RED_COLOR);
            pde_out_c("Quitting the iteration (the partial results will be written). "
                << newton_iter, RED_COLOR);
            pde_out_c("Originally: " << e.what(), RED_COLOR);
            break;
        }
        pde_out_c("Complete after " << solver_control.last_step() << " GMRES iterations", RED_COLOR);
        pde_out_c("L2 Norm of the step: " << step.l2_norm(),                              RED_COLOR);

        // Apply the damping coefficient
        solution.add(-alpha_k, step);
        // At each step, ensure the solution has the right constraints. Use the
        // object oriented version to just limit the z component of the displacement
        constraints.distribute(solution);

        for (int inaccurate_newton = 0; inaccurate_newton < 5; ++inaccurate_newton) {

            // Solve a new instance with the old jacobian, naturally weight this much less then
            // a correct jacobian iteration
            this->build_system( key("Build jacobian = ", false) );
            pde_out_c("L2 Norm of corrected residual: " << nr_rhs_f.l2_norm(), RED_COLOR);

            try {
                solver.solve(jacobian, step, nr_rhs_f, precondition);
            }
            catch (const SolverControl::NoConvergence& e) {
                pde_out_c("FAILED to solve the inaccurate GMRES iteration at step " << newton_iter, RED_COLOR);
                pde_out_c("Originally: " << e.what(), RED_COLOR);
                break;
            }

            pde_out_c("Complete after " << solver_control.last_step() << " GMRES iterations", RED_COLOR);
            pde_out_c("L2 Norm of the step: " << step.l2_norm(), RED_COLOR);

            // Apply the damping coefficient
            solution.add(-alpha_k / 3.5 , step);
        }

    }
    pde_out_c("Completed the newton iteration, saving the result",  GRN_COLOR);
    DataOut<dim> data_out;

    data_out.add_data_vector(dof_handler, solution, "solution");
    data_out.build_patches();

    const std::string output_file_name = "ortho_z_6.vtk";
    std::ofstream output_file(output_file_name);
    data_out.write_vtk(output_file);
    pde_out_c("Result saved into file " << output_file_name,        GRN_COLOR);

    return;
}

void
SuperElasticOrthotropicSolver::voigt_apply_to(
    const Tensor<2, dim>& F, const Tensor <2, dim>& tensor, Tensor<2, dim>& into,
    const pass_cache_data_t& intermediate) {
    Tensor<2, dim> t{ };
    Tensor<2, dim> m{ };
    Tensor<2, dim> reuse{ };

    // FullMatrix<double> voigt_matrix(9, 9);
    // Vector<double> voigt_vec(9);
    // Vector<double> out(9);

    const auto F_inv = invert(F);
    const auto Fmt = transpose(F_inv);
    const auto J = determinant(F);
    const auto Jm2o3 = pow_m2t(J);

    // pde_out_c("Determinant " << Jm2o3, RED_COLOR);
    // Here we construct the voigt matrix
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {

            // Note: this terms arise from  a naive "guccione" law
            /*
            t = F;
            t -= (1/ 3.0) * Fmt;
            t *= Jm2o3 * 2 * b;
            t *= (F[i][j] - (1 / 3.0) * Fmt[i][j] * intermediate.I1 - (2 / 3.0) * Fmt[i][j]);

            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    // Notice: we transpose in place by constructing into m[l][k] instead of 
                    // m[k][l]
                    m[l][k] = -F_inv[k][i] * F_inv[j][l];

            t[i][j] += 1;
            t -= (1 / 3.0) * m;
            t *= a * intermediate.exp_bi1m3 * Jm2o3;
            */
            
            t = F;
            t -= (1 / 3.0) * Fmt;
            t *= b * ( (-2/3.0) * intermediate.I1 * Fmt[i][j] + 2*F[i][j]*Jm2o3) - (2 / 3.0) * Fmt[i][j];
            //  t *= Jm2o3 * 2 * b;
            //  t *= (F[i][j] - (1 / 3.0) * Fmt[i][j] * intermediate.I1 - (2 / 3.0) * Fmt[i][j]);

            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    // Notice: we transpose in place by constructing into m[l][k] instead of
                    // m[k][l]
                    m[l][k] = -F_inv[k][i] * F_inv[j][l];

            t[i][j] += 1;
            t -= (1 / 3.0) * m;
            t *= a * intermediate.exp_bi1m3 * Jm2o3;

            // Now account for bulk modulus terms..

            // t += (bulk)*J * J * Fmt;
            // t += (bulk / 2) * (J * J - 1) * m;
   /*
            // Now account for orthotropy terms...

            if (intermediate.i4f > 0)
                t += 2 * af * intermediate.ff0t * intermediate.ff0t;
            reuse = 0.0;
            for (int k = 0; k < 3; ++k)
                reuse[i][k] = intermediate.ff0t[j][k];
            t += 2 * af * macaulay(intermediate.i4f) * reuse;


            if (intermediate.i4s > 0)
                t += 2 * as * intermediate.ss0t * intermediate.ss0t;
            reuse = 0.0;
            for (int k = 0; k < 3; ++k)
                reuse[i][k] = intermediate.ss0t[j][k];
            t += 2 * as * macaulay(intermediate.i4s) * reuse;
            */
            into[i][j] = scalar_product(t, tensor);
            // const auto bf = t.begin_raw();
            // for (int u = 0; u < 9; ++u)
            //    voigt_matrix[3 * i + j][u] = bf[u];

        }

#define transfer(expr, into) for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) expr = into;

    // WE can no use the voigt matri xcomptued at time step k to compute the 
    // actuve cirrectuib ti tge voigt matrix

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            // Initialize data to 

            for (int l = 0; l < 3; ++l) {

            }
        }
    // transfer(voigt_vec[3 * i + j], tensor[i][j]);

    // voigt_matrix.vmult(out, voigt_vec);
    // transfer(into[i][j], out[3 * i + j]);

#undef transfer
    return;
}



void
SuperElasticOrthotropicSolver::compute_deP_deF_at_q(
    const Tensor<2, dim>& F,
    const pass_cache_data_t& intermediate
) {
    Tensor<2, dim> t{ };
    Tensor<2, dim> m{ };
    Tensor<2, dim> reuse{ };

    const auto& F_inv   = intermediate.Finv;
    const auto& Fmt     = intermediate.Fmt;
    const auto J        = intermediate.J;
    const auto Jm2o3    = pow_m2t(J);
    const auto qform_f  = scalar_product(intermediate.f0, F * intermediate.f0);
    const auto qform_s  = scalar_product(intermediate.s0, F * intermediate.s0);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {

            t = 0.0;
            // Note: this terms arise from  a naive "guccione" law
            // t = 0.0;
            // t[i][j] = 2*a*1.0;
            
            /*
            t = F;
            t -= (1 / 3.0) * Fmt;
            t *= b * ( (-2/3.0) * intermediate.I1 * Fmt[i][j] + 2*F[i][j]*Jm2o3) - (2 / 3.0) * Fmt[i][j];
            //  t *= Jm2o3 * 2 * b;
            //  t *= (F[i][j] - (1 / 3.0) * Fmt[i][j] * intermediate.I1 - (2 / 3.0) * Fmt[i][j]);

            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    // Notice: we transpose in place by constructing into m[l][k] instead of 
                    // m[k][l]
                    m[l][k] = -F_inv[k][i] * F_inv[j][l];

            t[i][j] += 1;
            t -= (1 / 3.0) * m;
            t *= a * intermediate.exp_bi1m3 * Jm2o3;
            */

            // Now account for bulk modulus terms..
            reuse = 0.0;
            reuse[i][j];
            t += (bulk) * J * J * Fmt * Fmt[i][j];
            t += (bulk / 2) * (J * J - 1) * transpose(-F_inv * reuse * F_inv);
            
            // Now account for orthotropy terms...

            /*
            if (std::abs(intermediate.i4f - 1) > 1e-8) {

                
                pde_out_c_var(intermediate.ff0t, RED_COLOR);
                pde_out_c_var(intermediate.f0, RED_COLOR);
                pde_out_c_var(intermediate.exp_mac_f_sq, RED_COLOR);
                pde_out_c_var(intermediate.i4f, RED_COLOR);
                pde_out_c_var(bf, RED_COLOR);
                pde_out_c_var(af, RED_COLOR);

                m = 4 * bf * (intermediate.i4f - 1) * (intermediate.i4f - 1) * 
                    qform_f * intermediate.ff0t;
                
                m += 2 * qform_f * intermediate.ff0t;

                reuse = 0.0;
                reuse[i][j] = 1.0;
                m += (intermediate.i4f - 1) * reuse * outer_product(intermediate.f0, intermediate.f0);


                m *= 2 * af * intermediate.exp_mac_f_sq;
                t += m;
            }
            */

            /*
            if (intermediate.i4s > 1) {
                m = 2 * bs * (intermediate.i4s - 1) * (intermediate.i4s - 1) * intermediate.ss0t * intermediate.ss0t[i][j];
                m += 2 * intermediate.ss0t * intermediate.ss0t[i][j];
                reuse = 0.0;
                for (int k = 0; k < 3; ++k)
                    reuse[i][k] = intermediate.s0s0t[j][k];
                m += (intermediate.i4s - 1) * reuse;
                m *= 2 * as * intermediate.exp_mac_s_sq;
                t += m;
            }
            */

            deP_deF_at_q[i * 3 + j] = t;

        }

    return;
}

void
SuperElasticOrthotropicSolver::voigt_apply_to_batch(
    const Tensor<2, dim>& F, const Tensor <2, dim>& left,
    const std::vector<Tensor<2, dim>>& right,
    const pass_cache_data_t& intermediate,
    Vector<double>& save_into
    ) {
    // computes for a batch k=1...N left : ( depdef * right_k )
    Tensor<2, dim> t{ };
    Tensor<2, dim> m{ };
    Tensor<2, dim> reuse{ };

    // FullMatrix<double> voigt_matrix(9, 9);
    // Vector<double> voigt_vec(9);
    // Vector<double> out(9);

    const auto F_inv = invert(F);
    const auto Fmt = transpose(F_inv);
    const auto J = determinant(F);
    const auto Jm2o3 = pow_m2t(J);

    // pde_out_c("Determinant " << Jm2o3, RED_COLOR);
    // Here we construct the voigt matrix

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {

            // Note: this terms arise from  a naive "guccione" law
            t = F;
            t -= (1 / 3.0) * Fmt;
            t *= Jm2o3 * 2 * b;
            t *= (F[i][j] - (1 / 3.0) * Fmt[i][j] * intermediate.I1 - (2 / 3.0) * Fmt[i][j]);

            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    // Notice: we transpose in place by constructing into m[l][k] instead of 
                    // m[k][l]
                    m[l][k] = -F_inv[k][i] * F_inv[j][l];

            t[i][j] += 1;
            t -= (1 / 3.0) * m;
            t *= a * intermediate.exp_bi1m3 * Jm2o3;

            // Now account for bulk modulus terms..

            t += (bulk)*J * J * Fmt;
            t += (bulk / 2) * (J * J - 1) * m;
            /*
                     // Now account for orthotropy terms...

                     if (intermediate.i4f > 0)
                         t += 2 * af * intermediate.ff0t * intermediate.ff0t;
                     reuse = 0.0;
                     for (int k = 0; k < 3; ++k)
                         reuse[i][k] = intermediate.ff0t[j][k];
                     t += 2 * af * macaulay(intermediate.i4f) * reuse;


                     if (intermediate.i4s > 0)
                         t += 2 * as * intermediate.ss0t * intermediate.ss0t;
                     reuse = 0.0;
                     for (int k = 0; k < 3; ++k)
                         reuse[i][k] = intermediate.ss0t[j][k];
                     t += 2 * as * macaulay(intermediate.i4s) * reuse;
                     */


            for (unsigned int k_batch = 0; k_batch < right.size(); ++k_batch) {
                // element i_j of k-th result to be multiplied with right[i][j]
                save_into[k_batch] += scalar_product(t, right[k_batch]) * left[i][j];
            }

        }

    return;
}


void
SuperElasticOrthotropicSolver::compute_basis_at_quadrature(
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
SuperElasticOrthotropicSolver::orthothropic_base_at(
    const dealii::Point<dim>& p, std::vector<dealii::Tensor<1, dim>>& basis, bool compute_n = false
) {
    const double x = p[0],   y = p[1],   z = p[2];

    const double csquared = MESH_ELLIPSOID_Z_DEFORMATION * MESH_ELLIPSOID_Z_DEFORMATION;
    const double cfourthp = csquared * csquared;

    const double r = std::sqrt(x * x + y * y + z * z / cfourthp);
    const double rr = std::sqrt(x * x + y * y);
    // Let s be the radial vector aligned with the collagen sheet. 
    auto& s = basis[0];
    s[0] = x / r; s[1] = y / r; s[2] = (z / csquared) / r;
    // FLAG[GEO] 
    double corrected_radius = std::sqrt(x * x + y * y + z * z / csquared);
    double r_over_width = (corrected_radius - MESH_ELLIPSOID_SMALL_RADIUS) / (MESH_ELLIPSOID_LARGE_RADIUS - MESH_ELLIPSOID_SMALL_RADIUS);
    r_over_width = (r_over_width < 0) ? 0 : (r_over_width > 1) ? 1 : r_over_width;

    double rads = (r_over_width - 0.5) * (r_over_width - 0.5) * (r_over_width - 0.5) * 8 * 1.03;
    rads = (r_over_width - 0.5) * 2 * 1.03;

    auto& f = basis[1];
    f[0] = -y / rr; f[1] = x / rr; f[2] = 0;

    auto& n = basis[2];
    n = cross_product_3d(f, s);

    // Rotate F in the tangent plane by the transmurally-variant degree
    f = f * std::cos(rads) + n * std::sin(rads);
    if (compute_n)
        n = cross_product_3d(s, f);

}

void 
SuperElasticOrthotropicSolver::build_system(bool build_jacobian) {


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
    Vector<double>     contribution_voigt_accumulator(dofs_per_cell);
    // Local contribution matrix to the rhs of the newton raphson iteration step
    Vector<double>     cell_nr_rhs(dofs_per_cell);

    // We will use this vector to store the global indices of the DoFs of the
    // current element within the loop.
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    std::vector<Tensor<2, dim>> shape_grad_ref(dofs_per_cell);

    std::vector<Tensor< 2, dim>> grad_u_q(fe_values.n_quadrature_points);
    std::vector<Tensor< 2, dim>> grad_u_q_surf(fe_face_values.n_quadrature_points);
    std::vector<std::vector<Tensor<1, dim>>> orth_u_q(
        fe_values.n_quadrature_points,
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

    alignas(cache_line_size()) pass_cache_data_t intermediate{ };

    // Zero out all the matrices being filled
    if (build_jacobian)
        jacobian = 0.0;
    nr_rhs_f = 0.0;

    // Progress bar index (utility)
    int prog_i   = 0;
    const auto n = this->mesh.n_active_cells();

    Tensor<2, dim> voigt_product;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        // Print a progress bar every 1% of the dofs handleld
        prog_bar_c(prog_i, n, RED_COLOR); ++prog_i;

        // Note: avoid evaluating the surface values here as most tetrahedrons
        // are not part of the surface, evaluate only inside the ifs.
        fe_values.reinit(cell);

        cell_j_matrix = 0.0;
        cell_nr_rhs = 0.0;

        fe_values[displacement].get_function_gradients(
            solution,     // The global solution vector
            grad_u_q      // Output: The calculated gradients at all q-points
        );

        // Here we iterate over *local* DoF indices.
        for (std::size_t i = 0; i < grad_u_q.size(); ++i) {
            // Compute F = I + grad u
            auto& t = grad_u_q[i];
            t[0][0] += 1; t[1][1] += 1; t[2][2] += 1;
        }

        const std::vector<Point<dim>>& pts = fe_values.get_quadrature_points();

        compute_basis_at_quadrature(
            pts,
            orth_u_q, key("Compute n0 vector", false)
        );

        for (unsigned int q = 0; q < n_q; ++q)
        {
            /*
            
            
            
            */
            const auto& F = grad_u_q[q];        // Simple reference for readibility of math
            const auto& s0 = orth_u_q[q][0];    // Reference for the sheet 
            const auto& f0 = orth_u_q[q][1];    // Reference for the fibers 

            // Here we assemble the local contribution for current cell and
            // current quadrature point, filling the local matrix and vector.
            const auto C = transpose(F) * F;

            compute_and_cache( i4f, scalar_product(f0, C * f0), intermediate );
            compute_and_cache( i4s, scalar_product(s0, C * s0), intermediate );

            compute_and_cache( exp_mac_s_sq, std::exp(bs * macaulay(i4s - 1) * macaulay(i4s - 1)), intermediate );
            compute_and_cache( exp_mac_f_sq, std::exp(bf * macaulay(i4f - 1) * macaulay(i4f - 1)), intermediate );

            // Notice that the outer product is akin to the tensor product of the two vectors
            // outer_product() fully unrolls the computation loop 
            compute_and_cache( ff0t, outer_product(F * f0, f0), intermediate );
            compute_and_cache( ss0t, outer_product(F * s0, s0), intermediate );

            intermediate.f0 = f0;
            intermediate.s0 = s0;

            compute_and_cache( J, determinant(F), intermediate );
            // We cache the invert explicitly (inspection of tensor.h lines 2850 give ca. 50 
            // memory accesses per inversion of matrix vs. a single cached access
            compute_and_cache( Finv, invert(F), intermediate );
            compute_and_cache( Fmt, transpose(Finv), intermediate );
            // Note: avoid caching J^-2/3, the pade approximation is fast enough and avoids
            // extra memory reads. And no need to cache I1 either since its only useful to 
            // compute exp_bi1m3
            compute_and_cache( I1, pow_m2t(J) * trace (C), intermediate );
            compute_and_cache( exp_bi1m3, std::exp(b * (I1 - 3)), intermediate );

            compute_deP_deF_at_q(F, intermediate);

            // No need to cache the maucalay bracket, it is inlined as a trivial if 

            for (const unsigned int i : fe_values.dof_indices()) {

                const Tensor<2, dim> grad_i = fe_values[displacement].gradient(i, q);

                /* -------------- Note ----------------
                * A previous version of this iteration computed the products in-place, 
                * avoiding the storage of the partial derivative but with an extra overhead.
                >    contribution_voigt_accumulator = 0.0;
                >    shape_grad_ref.clear(); // Clear the copies
                >    for (const unsigned int j : fe_values.dof_indices())
                >        shape_grad_ref.emplace_back(fe_values[displacement].gradient(j, q));
                >    voigt_apply_to_batch(F, grad_i, shape_grad_ref, intermediate, contribution_voigt_accumulator);
                >    for (const unsigned int j : fe_values.dof_indices()) {
                >        cell_j_matrix(i, j) += fe_values.JxW(q) * contribution_voigt_accumulator[j];
                >    }
                */

                if (build_jacobian) {

                    const Tensor<2, dim> grad_i = fe_values[displacement].gradient(i, q);

                    for (const unsigned int j : fe_values.dof_indices()) {
                        const Tensor<2, dim> grad_j = fe_values[displacement].gradient(j, q);

                        cell_j_matrix(i, j) += fe_values.JxW(q) * scalar_product(
                            grad_i, tensor_product(deP_deF_at_q, grad_j)
                        );

                    }
                    /*
                    for (const unsigned int j : fe_values.dof_indices()) {

                        const Tensor<2, dim> grad_j = fe_values[displacement].gradient(j, q);
                        Tensor<2, dim> voigt_product;

                        voigt_apply_to(F, grad_j, voigt_product, intermediate);

                        cell_j_matrix(i, j) += fe_values.JxW(q) *
                            scalar_product(grad_i,
                                
                                // ====================== fully neo-hookean =========================
                                // (mu * grad_j) 
                                voigt_product
                            );
                    }
                    */
                }
                double guc_modified_i1_contribution = scalar_product(
                    a * pow_m2t(J) * exp_bi1m3 * (F - (1 / 3.0) * Fmt)
                    , grad_i) * fe_values.JxW(q);
                
                double bulk_modulus_contribution = scalar_product(
                    (bulk / 2) * (J * J - 1) * Fmt
                    , grad_i) * fe_values.JxW(q);

                /*
                double orthotropy_s0, orthotropy_f0;
                if (std::abs(i4s - 1) > 1e-8)
                    orthotropy_s0 = 2 * as * (i4s - 1) * exp_mac_s_sq * scalar_product(
                        ff0t, grad_i) * fe_values.JxW(q);
                else orthotropy_s0 = 0;

                if (std::abs(i4f - 1) > 1e-8)
                    orthotropy_f0 = 2 * af * (i4f - 1) * exp_mac_f_sq * scalar_product(
                        ss0t, grad_i) * fe_values.JxW(q);
                else orthotropy_f0 = 0;

                // cell_nr_rhs(i) += orthotropy_s0;
                // cell_nr_rhs(i) += orthotropy_f0;
                */

                cell_nr_rhs(i) += bulk_modulus_contribution;
                // cell_nr_rhs(i) += guc_modified_i1_contribution;
                // cell_nr_rhs(i) += 2 * a * scalar_product(F, grad_i) * fe_values.JxW(q);
            }
        }

        // NOTE: the computation of the boundary conditions does not
        // depend on the physical problem itself, just on the stress tensor J
        // and its cofactor.
        for (unsigned int face_no = 0; face_no < cell->reference_cell().n_faces(); ++face_no) {
            // Check if this face is a boundary
            const auto id = cell->face(face_no)->boundary_id();
            if (cell->face(face_no)->at_boundary() && (is_neumann(id) || is_robin(id))) {
                fe_face_values.reinit(cell, face_no);


                if (is_robin(id)) {

                    fe_face_values[displacement].get_function_values(
                        solution,     // The global solution vector
                        val_u_q_surf      // Output: The calculated gradients at all q-points
                    );

                    for (unsigned int q = 0; q < bdn_q; ++q)
                    {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                            const auto phi_i = fe_face_values[displacement].value(i, q);

                            if (build_jacobian)
                                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                                    const auto phi_j = fe_face_values[displacement].value(j, q);

                                    cell_j_matrix(i, j) += fe_values.JxW(q) * scalar_product(phi_i, alfa * phi_j);
                                }

                            // Rhs contribution

                            double up = alfa * scalar_product(
                                phi_i, val_u_q_surf[q]
                            ) * fe_values.JxW(q);
                            cell_nr_rhs(i) += up;
                        }
                    }

                }
                // IMPORTANT: This reinitializes the mapping for the current 2D face in 3D space.
                else if (is_neumann(id)) {

                    cof_f_q_surf.clear();
                    fe_face_values[displacement].get_function_gradients(
                        solution,
                        grad_u_q_surf
                    );

                    for (std::size_t i = 0; i < grad_u_q_surf.size(); ++i) {
                        // Compute F = I + grad u
                        auto& t = grad_u_q_surf[i];
                        t[0][0] += 1; t[1][1] += 1; t[2][2] += 1;
                        // Compute F^-t
                        const auto& f_mt = transpose(invert(t));
                        cof_f_q_surf.emplace_back(f_mt);
                    }

                    // After the loop, the grad_u_q_surf vector was modified so that each
                    // element became f_q = I + grad u_q
                    auto& f_q_surf = grad_u_q_surf;

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
                                    cell_j_matrix(i, j) += fe_values.JxW(q) *
                                        scalar_product(phi_i, p_v * (
                                            scalar_product(cof_f_q_surf[q], grad_j) * dealii::Physics::Elasticity::StandardTensors<dim>::I +
                                            -cof_f_q_surf[q] * transpose(grad_j)
                                            ) * cofactor * normal_q
                                        );

                            }


                            // RHS contribution

                            double up = p_v * fe_values.JxW(q) * scalar_product(
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

        // constraints.distribute_local_to_global(cell_j_matrix, cell_nr_rhs, dof_indices, jacobian, nr_rhs_f);
        jacobian.add(dof_indices, cell_j_matrix);
        nr_rhs_f.add(dof_indices, cell_nr_rhs);

    }
    MatrixTools::apply_boundary_values(
        boundary_values, jacobian, step, nr_rhs_f, true);

    // Util macro... ends the progress bar
    end_prog_bar();

}

void
SuperElasticOrthotropicSolver::compute_rh_s_newt_raphs() {
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


    std::vector<Tensor< 2, dim>> grad_u_q(fe_values.n_quadrature_points);
    std::vector<Tensor< 2, dim>> grad_u_q_surf(fe_face_values.n_quadrature_points);
    std::vector<std::vector<Tensor<1, dim>>> orth_u_q(
        fe_values.n_quadrature_points,
        std::vector<Tensor< 1, dim>>(3)
    );

    std::vector<Tensor< 1, dim>> val_u_q_surf(fe_face_values.n_quadrature_points);
    std::vector<Tensor< 2, dim>> cof_f_q_surf(fe_face_values.n_quadrature_points);

    const FEValuesExtractors::Vector displacement(0);

    std::map<types::global_dof_index, double> boundary_values;

    pass_cache_data_t intermediate{ };


    // Zero out all the matrices being filled
    nr_rhs_f = 0.0;

    // Progress bar index (utility)
    int prog_i = 0;
    const auto n = this->mesh.n_active_cells();

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        // Print a progress bar every 1% of the dofs handleld
        prog_bar_c(prog_i, n, YEL_COLOR); ++prog_i;

        // Note: avoid evaluating the surface values here as most tetrahedrons
        // are not part of the surface, evaluate only inside the ifs.
        fe_values.reinit(cell);

        cell_j_matrix = 0.0;
        cell_nr_rhs = 0.0;

        fe_values[displacement].get_function_gradients(
            solution,     // The global solution vector
            grad_u_q      // Output: The calculated gradients at all q-points
        );

        // Here we iterate over *local* DoF indices.
        for (std::size_t i = 0; i < grad_u_q.size(); ++i) {
            // Compute F = I + grad u
            auto& t = grad_u_q[i];
            t[0][0] += 1;
            t[1][1] += 1;
            t[2][2] += 1;
            // std::cout << t[0][0] << " " << t[0][1] << " " << t[0][2] << std::endl;
            // pde_out(grad_u_q[i]);
        }

        const std::vector<Point<dim>>& pts = fe_values.get_quadrature_points();

        compute_basis_at_quadrature(
            pts,
            orth_u_q, false
        );


        for (unsigned int q = 0; q < n_q; ++q)
        {
            const auto& F = grad_u_q[q];        // Simple reference for readibility of math
            const auto& s0 = orth_u_q[q][0];    // Reference for the sheet 
            const auto& f0 = orth_u_q[q][1];    // Reference for the fibers 

            // Here we assemble the local contribution for current cell and
            // current quadrature point, filling the local matrix and vector.
            const auto C = transpose(F) * F;

            compute_and_cache(i4f, scalar_product(f0, C * f0), intermediate);
            compute_and_cache(i4s, scalar_product(s0, C * s0), intermediate);

            compute_and_cache(exp_mac_s_sq, std::exp(bs * macaulay(i4s - 1) * macaulay(i4s - 1)), intermediate);
            compute_and_cache(exp_mac_f_sq, std::exp(bf * macaulay(i4f - 1) * macaulay(i4f - 1)), intermediate);

            // Notice that the outer product is akin to the tensor product of the two vectors
            // outer_product() fully unrolls the computation loop 
            compute_and_cache(ff0t, outer_product(F * f0, f0), intermediate);
            compute_and_cache(ss0t, outer_product(F * s0, s0), intermediate);

            intermediate.f0 = f0;
            intermediate.s0 = s0;

            compute_and_cache(J, determinant(F), intermediate);
            // We cache the invert explicitly (inspection of tensor.h lines 2850 give ca. 50 
            // memory accesses per inversion of matrix vs. a single cached access
            compute_and_cache(Finv, invert(F), intermediate);
            compute_and_cache(Fmt, transpose(Finv), intermediate);
            // Note: avoid caching J^-2/3, the pade approximation is fast enough and avoids
            // extra memory reads. And no need to cache I1 either since its only useful to 
            // compute exp_bi1m3
            compute_and_cache(I1, pow_m2t(J) * trace(C), intermediate);
            compute_and_cache(exp_bi1m3, std::exp(b * (I1 - 3)), intermediate);

            for (const unsigned int i : fe_values.dof_indices()) {

                const Tensor<2, dim> grad_i = fe_values[displacement].gradient(i, q);

                const double J = determinant(F);
                const auto Fmt = transpose(invert(F));
                // small values of the determinant

                compute_and_cache( I1, trace(transpose(F) * F), intermediate );

                double guc_modified_i1_contribution = scalar_product(
                    /*
                    ====================== fully neo-hookean =========================
                    (mu * grad_u_q[q]) 
                    a * pow_m2t(J) * (grad_u_q[q] - (1 / 3.0) * Fmt)*/
                    a * pow_m2t(J) * exp_bi1m3 * (F - (1 / 3.0) * Fmt)
                    , grad_i) * fe_values.JxW(q);
                
                double bulk_modulus_contribution = scalar_product(
                    (bulk / 2) * (J * J - 1) * Fmt, grad_i) * fe_values.JxW(q);

                double orthotropy_s0, orthotropy_f0;
                if (std::abs(i4s - 1) > 1e-8)
                    orthotropy_s0 = 2 * as * (i4s - 1) * exp_mac_s_sq * scalar_product(
                        ff0t, grad_i) * fe_values.JxW(q);
                else orthotropy_s0 = 0;

                if (std::abs(i4f - 1) > 1e-8)
                    orthotropy_f0 = 2 * af * (i4f - 1) * exp_mac_f_sq * scalar_product(
                        ss0t, grad_i) * fe_values.JxW(q);
                else orthotropy_f0 = 0;

                // cell_nr_rhs(i) += orthotropy_s0;
                // cell_nr_rhs(i) += orthotropy_f0;               

                cell_nr_rhs(i) += bulk_modulus_contribution;
                cell_nr_rhs(i) += guc_modified_i1_contribution;
            }
        }

        for (unsigned int face_no = 0; face_no < cell->reference_cell().n_faces(); ++face_no) {
            // Check if this face is a boundary
            const auto id = cell->face(face_no)->boundary_id();
            if (cell->face(face_no)->at_boundary() && (is_neumann(id) || is_robin(id))) {
                fe_face_values.reinit(cell, face_no);


                if (is_robin(id)) {

                    fe_face_values[displacement].get_function_values(
                        solution,     // The global solution vector
                        val_u_q_surf      // Output: The calculated gradients at all q-points
                    );

                    for (unsigned int q = 0; q < bdn_q; ++q)
                    {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                            const auto phi_i = fe_face_values[displacement].value(i, q);

                            double up = alfa * scalar_product(
                                phi_i, val_u_q_surf[q]
                            ) * fe_values.JxW(q);
                            cell_nr_rhs(i) += up;
                        }
                    }

                }
                // IMPORTANT: This reinitializes the mapping for the current 2D face in 3D space.
                else if (is_neumann(id)) {

                    cof_f_q_surf.clear();
                    fe_face_values[displacement].get_function_gradients(
                        solution,
                        grad_u_q_surf
                    );

                    for (std::size_t i = 0; i < grad_u_q_surf.size(); ++i) {
                        // Compute F = I + grad u
                        auto& t = grad_u_q_surf[i];
                        t[0][0] += 1; t[1][1] += 1; t[2][2] += 1;
                        // Compute F^-t
                        const auto& f_mt = transpose(invert(t));
                        cof_f_q_surf.emplace_back(f_mt);
                    }

                    auto& f_q_surf = grad_u_q_surf;
                    for (unsigned int q = 0; q < bdn_q; ++q)
                    {
                        const auto normal_q = fe_face_values.normal_vector(q);
                        const auto d = determinant(f_q_surf[q]);
                        const auto cofactor = cof_f_q_surf[q] * d;

                        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                            const auto phi_i = fe_face_values[displacement].value(i, q);
                            double up = p_v * fe_values.JxW(q) * scalar_product(
                                phi_i, cofactor * normal_q
                            );
                            cell_nr_rhs(i) += up;
                        }


                    }
                }
            }
        }

        cell->get_dof_indices(dof_indices);
        constraints.distribute_local_to_global(cell_nr_rhs, dof_indices, nr_rhs_f);
    }

    // Util macro... ends the progress bar
    end_prog_bar();
}


#endif