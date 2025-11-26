
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/fe_system.h>

#include <fstream>
#include <iostream>

#include "superelastic_isotropic.hpp"
#include "boundaries.hpp"

// Uncomment this definition to get verbose output (datetime and code line)
#define PDE_OUT_VERBOSE
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
        nr_rhs_f.reinit(dof_handler.n_dofs());
    }
}

void
SuperElasticIsotropicSolver::solve() {


    pde_out_c("Assembling the linear system", RED_COLOR);

    // Number of local DoFs for each element.
    const unsigned int dofs_per_cell = fe->dofs_per_cell;

    pde_out_c("DOFS per Cell" << dofs_per_cell, RED_COLOR);

    // Number of quadrature points for each element.
    const unsigned int n_q = quadrature->size();
    const unsigned int bdn_q = quadrature->size();

    
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

    jacobian = 0.0;
    nr_rhs_f = 0.0;

    std::vector< 
        std::vector< Tensor< 1, dim> >
    > grad_u_q(fe_values.n_quadrature_points);
    for (auto& un_vec : grad_u_q)
        un_vec = std::vector< Tensor< 1, dim> >(dim);
    solution = 0.15;

    const FEValuesExtractors::Vector velocities(0);

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);

        cell_j_matrix = 0.0;
        cell_nr_rhs = 0.0;
        /*
        // 2. Single function call to calculate all gradients at once.
        fe_values.get_function_gradients(
            solution,     // The global solution vector
            grad_u_q      // Output: The calculated gradients at all q-points
        );*/

        for (unsigned int q = 0; q < n_q; ++q)
        {
            // Here we assemble the local contribution for current cell and
            // current quadrature point, filling the local matrix and vector.

            // Here we iterate over *local* DoF indices.
            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    
                    pde_out_c("Computing gradient", RED_COLOR);

                    Tensor<2, dim> t1 = fe_values[velocities].gradient(i, q);
                    Tensor<2, dim> t2 = fe_values[velocities].gradient(j, q);
                    pde_out_c(t1, RED_COLOR);
                    // cell_j_matrix(i, j) += fe_values.JxW(q) *
                    //     double_contract<0, 1, 0, 1, 2, 2, dim, double, double>(t1, t2);
                }
                /*
                cell_rhs(i) += f_loc *                       //
                    fe_values.shape_value(i, q) * //
                    fe_values.JxW(q);*/
            }
        }
        pde_out_c("Handling surface boundaries", BLU_COLOR);
        for (unsigned int face_no = 0; face_no < cell->reference_cell().n_faces(); ++face_no) {
            // Check if this face is a boundary
            pde_out_c("GEOMETRY: FACES PER CELL" << GeometryInfo<dim>::faces_per_cell, RED_COLOR);
            if (cell->face(face_no)->at_boundary() &&
                is_a_boundary(cell->face(face_no)->boundary_id())) {
                // IMPORTANT: This reinitializes the mapping for the current 2D face in 3D space.
                fe_face_values.reinit(cell, face_no);

                for (unsigned int q = 0; q < bdn_q; ++q)
                {
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                        }
                    }
                }
                // ... now perform integration ...
            }
        }
    }
/*


        // Reinitialize the FEValues object on current element. This
        // precomputes all the quantities we requested when constructing
        // FEValues (see the update_* flags above) for all quadrature nodes of
        // the current cell.
        fe_values.reinit(cell);

        // We reset the cell matrix and vector (discarding any leftovers from
        // previous element).
        cell_matrix = 0.0;
        cell_rhs = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            // Here we assemble the local contribution for current cell and
            // current quadrature point, filling the local matrix and vector.
            const double mu_loc = mu(fe_values.quadrature_point(q));
            const double f_loc = f(fe_values.quadrature_point(q));

            // Here we iterate over *local* DoF indices.
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cell_matrix(i, j) += mu_loc *                     //
                        fe_values.shape_grad(i, q) * //
                        fe_values.shape_grad(j, q) * //
                        fe_values.JxW(q);
                }

                cell_rhs(i) += f_loc *                       //
                    fe_values.shape_value(i, q) * //
                    fe_values.JxW(q);
            }
        }

        // At this point the local matrix and vector are constructed: we need
        // to sum them into the global matrix and vector. To this end, we need
        // to retrieve the global indices of the DoFs of current cell.
        cell->get_dof_indices(dof_indices);

        // Then, we add the local matrix and vector into the corresponding
        // positions of the global matrix and vector.
        system_matrix.add(dof_indices, cell_matrix);
        system_rhs.add(dof_indices, cell_rhs);
    }

    // Boundary conditions.
    //
    // So far we assembled the matrix as if there were no Dirichlet conditions.
    // Now we want to replace the rows associated to nodes on which Dirichlet
    // conditions are applied with equations like u_i = b_i. We use deal.ii
    // functions to
    {
        // We construct a map that stores, for each DoF corresponding to a Dirichlet
        // condition, the corresponding value. E.g., if the Dirichlet condition is
        // u_i = b_i, the map will contain the pair (i, b_i).
        std::map<types::global_dof_index, double> boundary_values;

        // This object represents our boundary data as a real-valued function (that
        // always evaluates to zero). Other functions may require to implement a
        // custom class derived from dealii::Function<dim>.
        Functions::ZeroFunction<dim> bc_function;

        // Then, we build a map that, for each boundary tag, stores a pointer to the
        // corresponding boundary function.
        std::map<types::boundary_id, const Function<dim>*> boundary_functions;
        boundary_functions[0] = &bc_function;
        boundary_functions[1] = &bc_function;

        // interpolate_boundary_values fills the boundary_values map.
        VectorTools::interpolate_boundary_values(dof_handler,
            boundary_functions,
            boundary_values);

        // Finally, we modify the linear system to apply the boundary conditions.
        // This replaces the equations for the boundary DoFs with the corresponding
        // u_i = 0 equations.
        MatrixTools::apply_boundary_values(
            boundary_values, system_matrix, solution, system_rhs, true);
    }
    */

}