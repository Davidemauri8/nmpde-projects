
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>


#include <fstream>
#include <iostream>

#include "superelastic_isotropic.hpp"
#include "boundaries.hpp"

// Uncomment this definition to get verbose output (datetime and code line)
#define PDE_OUT_VERBOSE
#define PDE_PROGRESS_BAR

#include "../utilities/mesh_io.hpp"

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

        pde_out_c("Initializing the sparsity pattern", GRN_COLOR);
        DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        pde_out_c("Copying the sparsity pattern", GRN_COLOR);
        sparsity_pattern.copy_from(dsp);
        jacobian.reinit(sparsity_pattern);

        pde_out_c_i("Initializing the solution vector", GRN_COLOR, 1);
        solution.reinit(dof_handler.n_dofs());
        step.reinit(dof_handler.n_dofs());
        nr_rhs_f.reinit(dof_handler.n_dofs());
    }
}

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

    solution = 0.0;

    for (int ITER_OUT = 0; ITER_OUT < 20; ++ITER_OUT) {

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

            const double mu = 0.15;
            // Here we iterate over *local* DoF indices.
            for (std::size_t i = 0; i < grad_u_q.size(); ++i) {
                // Compute F = I + grad u
                auto& t = grad_u_q[i];
                t[0][0] += 1;
                t[1][1] += 1;
                t[2][2] += 1;
            }

            for (unsigned int q = 0; q < n_q; ++q)
            {
                // Here we assemble the local contribution for current cell and
                // current quadrature point, filling the local matrix and vector.


                for (const unsigned int i : fe_values.dof_indices()) {
                    const Tensor<2, dim> grad_i = fe_values[displacement].gradient(i, q);

                    for (const unsigned int j : fe_values.dof_indices()) {

                        const Tensor<2, dim> grad_j = fe_values[displacement].gradient(j, q);

                        cell_j_matrix(i, j) += fe_values.JxW(q) *
                            scalar_product(grad_i, (mu * grad_j));
                    }

                    cell_nr_rhs(i) += scalar_product(mu * grad_u_q[q], grad_i) * fe_values.JxW(q);
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
                    
                        for (unsigned int q = 0; q < bdn_q; ++q)
                        {
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                                    const auto phi_i = fe_face_values[displacement].value(i, q);
                                    const auto phi_j = fe_face_values[displacement].value(j, q);

                                    cell_j_matrix(i, j) += fe_values.JxW(q) * scalar_product(phi_i, alfa * phi_j);
                                }
                        }

                    
                    
                    }
                    // IMPORTANT: This reinitializes the mapping for the current 2D face in 3D space.
                    else if (is_neumann(id)) {

                        fe_face_values[displacement].get_function_gradients(
                            solution,
                            grad_u_q_surf
                        );

                        for (std::size_t i = 0; i < grad_u_q_surf.size(); ++i) {
                            cof_f_q_surf.clear();
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
                            const auto cofactor = cof_f_q_surf[q] * determinant(f_q_surf[q]);
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
            jacobian.add(dof_indices, cell_j_matrix);
            nr_rhs_f.add(dof_indices, cell_nr_rhs);

        }
        end_prog_bar();


        ReductionControl solver_control(/* maxiter = */ 100,
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
        MatrixTools::apply_boundary_values(
            boundary_values, jacobian, step, nr_rhs_f, false
        );
        
        pde_out_c("L2 Norm of residual: " << nr_rhs_f.l2_norm(), RED_COLOR);

        PreconditionSSOR<SparseMatrix<double> > precondition;
        precondition.initialize(
            jacobian, PreconditionSSOR<SparseMatrix<double>>::AdditionalData(.6));

        solver.solve(jacobian, step, nr_rhs_f, precondition);
        pde_out_c(solver_control.last_step() << " GMRES iterations", RED_COLOR);

        pde_out_c("L2 Norm: " << step.l2_norm(), RED_COLOR);

        solution -= step;
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
        "outputmesh12.vtk";
    std::ofstream output_file(output_file_name);
    data_out.write_vtk(output_file);

}