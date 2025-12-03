
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

#include "superelastic_isotropic.hpp"
#include "mesh_geometry.hpp"
#include "boundaries.hpp"

// Uncomment this definition to get verbose output (datetime and code line)
#define PDE_OUT_VERBOSE
#define PDE_PROGRESS_BAR

#include "../utilities/visualize.hpp"
#include "../utilities/mesh_io.hpp"

// Model properties, TEMPORARY!
#define ACTIVE_FORMULATION

using namespace dealii;

void
SuperElasticIsotropicSolver::setup(const std::string& mesh_path) {
	
	// Load the mesh into the mesh item with an util function
	{
		pde_out_c("Loading mesh " << mesh_path, RED_COLOR);
		UtilsMesh::load_mesh_into_tria(mesh_path, this->mesh);
	}
    {
        pde_out_c("Initializing the finite element space", YEL_COLOR);

        // Finite elements in one dimension are obtained with the FE_Q or
        // FE_SimplexP classes (the former is meant for hexahedral elements, the
        // latter for tetrahedra, but they are equivalent in 1D). We use FE_SimplexP
        // here for consistency with the next labs.
        fe = std::make_unique<FESystem<dim>>(FE_SimplexP<dim>(this->r) ^ dim);

        pde_out_c_i("Degree = " << fe->degree, YEL_COLOR, 1);
        pde_out_c_i("DoFs per cell = " << fe->dofs_per_cell, YEL_COLOR, 1);

        // Construct the quadrature formula of the appopriate degree of exactness.
        // This formula integrates exactly the mass matrix terms (i.e. products of
        // basis functions).
        quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);
        surf_quadrature = std::make_unique<QGaussSimplex<dim-1>>(r + 1);

        pde_out_c_i("Volume quadrature points per cell = " << quadrature->size(), YEL_COLOR, 1);
        pde_out_c_i("Surface quadrature points per cell = " << surf_quadrature->size(), YEL_COLOR, 1);

    }

    // 
    {
        pde_out_c("Initializing the DoF handler", BLU_COLOR);

        // Initialize the DoF handler with the mesh we constructed.
        dof_handler.reinit(mesh);

        // "Distribute" the degrees of freedom. For a given finite element space,
        // initializes info on the control variables (how many they are, where
        // they are collocated, their "global indices", ...).
        dof_handler.distribute_dofs(*fe);
        pde_out_c_i("Number of DoFs = " << dof_handler.n_dofs(), BLU_COLOR, 1);
    }

    // Initialize the linear system.
    {

        constraints.clear();
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

        pde_out_c("Initializing the sparsity pattern", GRN_COLOR);
        DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

        DoFTools::make_sparsity_pattern(dof_handler, dsp
            
            , constraints, true);
        pde_out_c("Copying the sparsity pattern", GRN_COLOR);
        sparsity_pattern.copy_from(dsp);
        jacobian.reinit(sparsity_pattern);

        pde_out_c_i("Initializing the solution vector", GRN_COLOR, 1);
        solution.reinit(dof_handler.n_dofs());
        step.reinit(dof_handler.n_dofs());
        nr_rhs_f.reinit(dof_handler.n_dofs());
    }
}


#if defined(_MSC_VER)
// Microsoft Visual C++ Compiler
#define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
// GCC or Clang
#define FORCE_INLINE __attribute__((always_inline)) inline
#else
// Fallback for other compilers
#define FORCE_INLINE inline
#endif
FORCE_INLINE
double
pow_m2t(double v) {
    const double av = (v > 0) ? v : -v;
    const double k = (5 + av) / (1 + 5 * av);
    return (v > 0)? k : -k;
}


FORCE_INLINE
double
macaulay(double arg) {
    return (arg > 0.0) ? arg : 0.0;
}

template <unsigned int Dim>
FORCE_INLINE
Tensor<2, Dim>outer_product(const Tensor<1, Dim> v1, const Tensor<1, Dim> v2) {
    Tensor<2, Dim> t;
    t[0][0] = v1[0] * v2[0];
    t[0][1] = t[1][0] = v1[0] * v2[1];
    t[0][2] = t[2][0] = v1[0] * v2[2];
    t[1][1] = v1[1] * v2[1]; 
    t[1][2] = t[2][1] = v1[1] * v2[2];
    t[2][2] = v1[2] * v1[2];
    return t;
}

constexpr const auto dim = SuperElasticIsotropicSolver::dim;


#define cache_into(name, expression, inter) const auto name = expression; inter.name = name; 

void
SuperElasticIsotropicSolver::solve() {


    pde_out_c("Assembling the linear system", RED_COLOR);

    // Number of local DoFs for each element.
    const unsigned int dofs_per_cell = fe->dofs_per_cell;

    pde_out_c("DOFS per Cell " << dofs_per_cell, RED_COLOR);

    // Number of quadrature points for each element.
    const unsigned int n_q = quadrature->size();
    const unsigned int bdn_q = surf_quadrature->size();

    
    // FEValues instance. This object allows to compute basis functions, their
    // derivatives, the reference-to-current element mapping and its
    // derivatives on all quadrature points of all elements.
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
    {
        Functions::ZeroFunction<dim> bc_function(dim);
        std::map<types::boundary_id, const Function<dim>*> boundary_functions;
        boundary_functions[PDE_DIRICHLET] = &bc_function;

        VectorTools::interpolate_boundary_values(dof_handler,
            boundary_functions,
            boundary_values);
    }

    UtilsMesh::view_cartesian_coords(this->mesh, fe, quadrature, "field_new.vtu");
    UtilsMesh::visualize_grain_fibers(
        [this](
            const dealii::Point<dim>& p, std::vector<dealii::Tensor<1, dim>>& v
            ) {
                return this->orthothropic_base_at(p, v, true);
        },
        this->mesh, fe, quadrature, "orth_field_new.vtu");
    UtilsMesh::visualize_wall_depth(this->mesh, "depth.vtu");

    pass_cache_data_t intermediate{ };

    solution = 0.1;
    double maxj = 0;
#define MAX_ITER_AMT 8
    for (int ITER_OUT = 0; ITER_OUT < MAX_ITER_AMT; ++ITER_OUT) {

        pde_out_c("Step number " << ITER_OUT, YEL_COLOR);
        step     = 0.0;
        jacobian = 0.0;
        nr_rhs_f = 0.0;

        const auto n = this->mesh.n_active_cells();
        int prog_i = 0;
        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            // Print a progress bar every 1000 dofs handleld
            prog_bar_c(prog_i, n, RED_COLOR); ++prog_i;

            fe_values.reinit(cell);

            cell_j_matrix = 0.0;
            cell_nr_rhs = 0.0;

            // Note: avoid evaluating the surface values here as most tetrahedrons
            // are not part of the surface.
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
                // Here we assemble the local contribution for current cell and
                // current quadrature point, filling the local matrix and vector.
                const auto C = transpose(grad_u_q[q]) * grad_u_q[q];
                cache_into( i4f, scalar_product(orth_u_q[q][1], C * orth_u_q[q][1]), intermediate );
                cache_into( i4s, scalar_product(orth_u_q[q][0], C * orth_u_q[q][0]), intermediate );

                const auto ff0t = outer_product(grad_u_q[q] * orth_u_q[q][0], orth_u_q[q][0]); intermediate.ff0t = ff0t;
                const auto ss0t = outer_product(grad_u_q[q] * orth_u_q[q][1], orth_u_q[q][1]); intermediate.ss0t = ss0t;

                for (const unsigned int i : fe_values.dof_indices()) {

                    const Tensor<2, dim> grad_i = fe_values[displacement].gradient(i, q);

                    for (const unsigned int j : fe_values.dof_indices()) {

                        const Tensor<2, dim> grad_j = fe_values[displacement].gradient(j, q);
                        Tensor<2, dim> voigt_product;

                        voigt_apply_to(grad_u_q[q], grad_j, voigt_product, intermediate);
                        cell_j_matrix(i, j) += fe_values.JxW(q) *
                            scalar_product(grad_i,
                                /*
                                ====================== fully neo-hookean =========================
                                (mu * grad_j) */
                                voigt_product
                            );
                    }

                    const double J = determinant(grad_u_q[q]);
                    const auto Fmt = transpose(invert(grad_u_q[q]));

                    double guc_modified_i1_contribution = scalar_product(
                        /*
                        ====================== fully neo-hookean =========================
                        (mu * grad_u_q[q]) */
                        mu * pow_m2t(J) * (grad_u_q[q] - (1/3.0)* Fmt)
                        , grad_i) * fe_values.JxW(q);

                    double bulk_modulus_contribution = scalar_product(
                        (bulk / 2) * (J * J - 1) * Fmt, grad_i) *fe_values.JxW(q);

                    // Contribution of orthotropy 
                    double orthotropy_s0 = 2 * as * macaulay(i4s - 1) * scalar_product(
                        ff0t, grad_i) * fe_values.JxW(q);
                    double orthotropy_f0 = 2 * af * macaulay(i4f - 1) * scalar_product(
                        ss0t, grad_i) * fe_values.JxW(q);


                    cell_nr_rhs(i) += orthotropy_s0;
                    cell_nr_rhs(i) += orthotropy_f0;

                    cell_nr_rhs(i) += guc_modified_i1_contribution;
                    cell_nr_rhs(i) += bulk_modulus_contribution;
                }
            }

            // NOTE: the computation of the boundary conditions does not
            // depend on the physical problem itself, just on the stress tensor J
            // and its cofactor.
            for (unsigned int face_no = 0; face_no < cell->reference_cell().n_faces(); ++face_no) {
                // Check if this face is a boundary
                const auto id = cell->face(face_no)->boundary_id();
                if (cell->face(face_no)->at_boundary() && (is_neumann(id) || is_robin(id)) ) {
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
                            maxj = (d < maxj) ? d : maxj;
                            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                                const auto phi_i = fe_face_values[displacement].value(i, q);
                                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                                {
                                    const auto grad_j = fe_face_values[displacement].gradient(j, q);

                                    cell_j_matrix(i, j) += fe_values.JxW(q) *
                                        scalar_product(phi_i, ch_p * (
                                            scalar_product(cof_f_q_surf[q], grad_j) * dealii::Physics::Elasticity::StandardTensors<dim>::I +
                                            -cof_f_q_surf[q] * transpose(grad_j)
                                            ) * cofactor * normal_q
                                        );

                                }


                                // RHS contribution

                                double up = ch_p * fe_values.JxW(q) * scalar_product(
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

            // Then, we add the local matrix and vector into the corresponding
            // positions of the global matrix and vector.

            constraints.distribute_local_to_global(
                cell_j_matrix, cell_nr_rhs, dof_indices, jacobian, nr_rhs_f);
            // jacobian.add(dof_indices, cell_j_matrix);
            // nr_rhs_f.add(dof_indices, cell_nr_rhs);

        }
        end_prog_bar();


        ReductionControl solver_control(/* maxiter = */ 400,
            /* tolerance = */ 1.0e-16,
            /* reduce = */ 1.0e-6);

        // Since the system matrix is symmetric and positive definite, we solve the
        // system using the conjugate gradient method.
        SolverGMRES<Vector<double>> solver(solver_control);

        pde_out_c("Solving the linear system", RED_COLOR);
        // We use the identity preconditioner for now.

        // Finally, we modify the linear system to apply the boundary conditions.
        // This replaces the equations for the boundary DoFs with the corresponding
        // u_i = 0 equations.

        
        /* MatrixTools::apply_boundary_values(
            boundary_values, jacobian, step, nr_rhs_f, false
        );*/
        
        pde_out_c("L2 Norm of residual: " << nr_rhs_f.l2_norm(), RED_COLOR);
        pde_out_c("L2 Norm of Jac: " << jacobian.l1_norm(), RED_COLOR);


        SparseILU<double> precondition;
        precondition.initialize(
            jacobian, SparseILU<double>::AdditionalData(0, 2));
        try {
            solver.solve(jacobian, step, nr_rhs_f, precondition);
            pde_out_c(solver_control.last_step() << " GMRES iterations", RED_COLOR);
        }
        catch (...) {
            break;
        }
        pde_out_c("L2 Norm: " << step.l2_norm(), RED_COLOR);
        const double alpha_k = 0.5 + 0.5 * (ITER_OUT) / MAX_ITER_AMT;
        solution.add(-alpha_k, step);
        constraints.distribute(solution);

    }


    DataOut<dim> data_out;

    // It can write multiple variables (defined on the same mesh) to a single
    // file. Each of them can be added by calling add_data_vector, passing the
    // associated DoFHandler and a name.
    data_out.add_data_vector(dof_handler, solution, "solution");

    // Once all vectors have been inserted, call build_patches to finalize the
    // DataOut object, preparing it for writing to file.
    data_out.build_patches();

    // Then, use one of the many write_* methods to write the file in an
    // appropriate format.
    const std::string output_file_name =
        "ortho_z_2.vtk";
    std::ofstream output_file(output_file_name);
    data_out.write_vtk(output_file);

}

void
SuperElasticIsotropicSolver::voigt_apply_to(
    const Tensor<2, dim>& F, const Tensor <2, dim>& tensor, Tensor<2, dim>& into,
    const pass_cache_data_t& intermediate) {
    Tensor<2, dim> t{ };
    Tensor<2, dim> m{ };
    Tensor<2, dim> reuse{ };

    FullMatrix<double> voigt_matrix(9, 9);
    Vector<double> voigt_vec(9);
    Vector<double> out(9);

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
            t -= 0.33333333 * Fmt;
            t *= Jm2o3 * mu * -0.66666666 * F_inv[j][i];
            t[i][j] += Jm2o3*mu;

            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    // Notice: we transpose in place by constructing into m[l][k] instead of 
                    // m[k][j
                    m[l][k] = -F_inv[k][i] * F_inv[j][l];
            t += -mu * Jm2o3 * 0.33333333 * m;

            // Now account for bulk modulus terms..

            t += (bulk)*J * J * Fmt;
            t += (bulk / 2) * (J * J - 1) * m;

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

            const auto bf = t.begin_raw();
            for (int u = 0; u < 9; ++u)
                voigt_matrix[3 * i + j][u] = bf[u];

        }

#define transfer(expr, into) for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) expr = into;

#ifdef ACTIVE_FORMULATION
    // WE can no use the voigt matri xcomptued at time step k to compute the 
    // actuve cirrectuib ti tge voigt matrix

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            // Initialize data to 

            for (int l = 0; l < 3; ++l) {

            }
        }

#endif
    transfer(voigt_vec[3 * i + j], tensor[i][j]);

    voigt_matrix.vmult(out, voigt_vec);
    transfer(into[i][j], out[3 * i + j]);

#undef transfer
    return;
}

void
SuperElasticIsotropicSolver::compute_basis_at_quadrature(
    const std::vector<Point<dim>>& p,
    std::vector<std::vector<dealii::Tensor<1, dim>>>& orth_sys, 
    bool compute_n
) {
    for (unsigned int i = 0; i < p.size(); ++i) {
        orthothropic_base_at(p[i], orth_sys[i], compute_n);
    }
}

void 
SuperElasticIsotropicSolver::orthothropic_base_at(
    const dealii::Point<dim>& p, std::vector<dealii::Tensor<1, dim>>& basis, bool compute_n = false
) {
    const double x = p[0];
    const double y = p[1];
    const double z = p[2];

    const double csquared = MESH_ELLIPSOID_Z_DEFORMATION * MESH_ELLIPSOID_Z_DEFORMATION;
    const double cfourthp = csquared * csquared;

    const double r = std::sqrt(x * x + y * y + z * z / cfourthp);
    const double rr = std::sqrt(x * x + y * y);
    // Let s be the radial vector aligned with the collagen sheet. 
    auto& s = basis[0];
    s[0] = x / r; s[1] = y / r; s[2] = (z/csquared) / r;
    // FLAG[GEO] 
    double corrected_radius = std::sqrt(x * x + y * y + z * z / csquared);
    double r_over_width = (corrected_radius - MESH_ELLIPSOID_SMALL_RADIUS) / (MESH_ELLIPSOID_LARGE_RADIUS-MESH_ELLIPSOID_SMALL_RADIUS) ;
    r_over_width = (r_over_width < 0) ? 0 : (r_over_width > 1) ? 1 : r_over_width;

    double rads = (r_over_width - 0.5) * (r_over_width - 0.5) * (r_over_width - 0.5) * 8 * 1.03;
    rads = (r_over_width - 0.5) * 2 * 1.03;
   
    auto& f = basis[1];
    f[0] = -y / rr; f[1] = x / rr; f[2] = 0;

    auto& n = basis[2];
    n = cross_product_3d(f, s);

    // n[0] = x * z / (rr*r); n[0] = -y * z / (rr*r); n[2] = -(x * x + y * y) / (rr*r);
    f = f * std::cos(rads) + n * std::sin(rads);
    if (compute_n)
        n = cross_product_3d(s, f);

}

void
SuperElasticIsotropicSolver::compute_rh_s_newt_raphs(Vector<double>& put) {

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe, *quadrature,
        update_values | update_gradients | update_quadrature_points |
        update_JxW_values);

    Vector<double>     cell_nr_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
    std::vector<Tensor< 2, dim>> grad_u_q(fe_values.n_quadrature_points);

    const FEValuesExtractors::Vector displacement(0);

    put = 0.0;
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);

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

        for (unsigned int q = 0; q < n_q; ++q)
            for (const unsigned int i : fe_values.dof_indices()) {
                const Tensor<2, dim> grad_i = fe_values[displacement].gradient(i, q);
                cell_nr_rhs(i) += scalar_product(mu * grad_u_q[q], grad_i) * fe_values.JxW(q);
            }

        cell->get_dof_indices(dof_indices);

        // Then, we add the local matrix and vector into the corresponding
        // positions of the global matrix and vector.
        put.add(dof_indices, cell_nr_rhs);
    }

}
 