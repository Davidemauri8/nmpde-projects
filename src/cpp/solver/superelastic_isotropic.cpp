
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

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
        fe = std::make_unique<FE_SimplexP<dim>>(r);

        pde_out_c_i("Degree = " << fe->degree, YEL_COLOR, 1);
        pde_out_c_i("DoFs per cell = " << fe->dofs_per_cell, YEL_COLOR, 1);

        // Construct the quadrature formula of the appopriate degree of exactness.
        // This formula integrates exactly the mass matrix terms (i.e. products of
        // basis functions).
        quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

        pde_out_c_i("Quadrature points per cell = " << quadrature->size(), YEL_COLOR, 1);
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
        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);

        jacobian.reinit(sparsity_pattern);
        pde_out_c_i("Initializing the solution vector", GRN_COLOR, 1);
        solution.reinit(dof_handler.n_dofs());
    }
}

void
SuperElasticIsotropicSolver::solve() {




}