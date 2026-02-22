#include "BoundariesBarbarotta.hpp"
#include "Barbarotta.hpp"

#include <deal.II/base/function.h>
#include <deal.II/differentiation/ad.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out_faces.h>

#include "../utilities/mesh_io.hpp"
#include "../utilities/math.hpp"

#include <map>

// Defining the method to populate the struct
#define compute_and_cache(name, expression, inter)                             \
  const auto name = expression;                                                \
  inter.name = name;

using ADHelper = Differentiation::AD::ResidualLinearization<
    Differentiation::AD::NumberTypes::sacado_dfad, double>;
using ADNumber = typename ADHelper::ad_type;

void BarbarottaSolver::setup(const std::string &mesh_path) {
  // Create the mesh.
  {
    pde_out_c_par(pcout, "Opening the mesh", RED_COLOR);

    Triangulation<dim> mesh_serial;
    { UtilsMesh::load_mesh_into_tria(mesh_path, mesh_serial); }

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

    FE_SimplexP<dim> fe_scalar(r);
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

  {
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    constraints.close();
    pde_out_c_par(pcout, "  Constraints reinitialized ", RED_COLOR);
  }

    pcout << "-----------------------------------------------" << std::endl;


  // Initialize the linear system.
    {
    pde_out_c_par(pcout, "Initializing the linear system", RED_COLOR);

    pde_out_c_par(pcout, "  Initializing the sparsity pattern", RED_COLOR);

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity, constraints);

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

ADNumber BarbarottaSolver::active_phi(const ADNumber i4) {
  return Sn * (1 + beta * (std::sqrt(i4) - 1.0)) / (i4 * std::sqrt(i4));
}

void BarbarottaSolver::compute_P_at_q(const Tensor<2, dim, ADNumber> &F_q,
                                        const pass_cache_data_t &i) {

    Tensor<2, dim, ADNumber> Ff0f0t = F_q * i.f0f0t;
    Tensor<2, dim, ADNumber> Fs0s0t = F_q * i.s0s0t;

    Tensor<1, dim, ADNumber> Ff0 = F_q * f0;
    Tensor<1, dim, ADNumber> Fs0 = F_q * s0;

    Tensor<2, dim, ADNumber> fs0_plus_sf0 =
        outer_product(Ff0, s0) + outer_product(Fs0, f0);

    // Initialize P
    P.clear();

    // Volumetric contribution
    P = (bulk / 2.0) * (i.J * i.J - 1.0) * i.Fmt;

    // Isochoric isotropic part
    P += a * pow_m2t(i.J) * i.exp_bi1m3 * (F_q - (1.0 / 3.0) * i.Fmt);

    // Orthotropic fiber contribution
    if (i.macf > 0.0) {
      P += 2.0 * af * i.macf * i.exp_mac_f_sq * Ff0f0t;
    }

    // Orthotropic sheet contribution
    if (i.macs > 0.0) {
      P += 2.0 * as * i.macs * i.exp_mac_s_sq * Fs0s0t;
    }

    // Shear f-s coupling
    P += asf * i.i8sf * i.exp_bi8sf_sq * fs0_plus_sf0;

    // Active stress
    ADNumber phi = active_phi(i.i4f);

    P += Ff0f0t * phi;
  }


void BarbarottaSolver::assemble_system() {
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();
  const unsigned int n_q_face = quadrature_face->size();

  FEValues<dim> fe_values(*fe, *quadrature,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(*fe, *quadrature_face,
                                   update_values | update_gradients | update_normal_vectors |
                                       update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  jacobian = 0.0;
  nr_rhs_f = 0.0;

  // alignas(cache_line_size()) it does not work, how to solve??
  pass_cache_data_t intermediate{};

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned()) {
      continue;
    }

    fe_values.reinit(cell);
    cell->get_dof_indices(dof_indices);

    cell_matrix = 0.0;
    cell_rhs = 0.0;

      std::vector<Tensor<1, dim>> val_u_q_surf(
        fe_face_values.n_quadrature_points);

    FEValuesExtractors::Vector displacement(0);

    const unsigned int n_independent_variables = dof_indices.size();
    const unsigned int n_dependent_variables = dofs_per_cell;
    ADHelper ad_helper(n_independent_variables, n_dependent_variables);

    ad_helper.register_dof_values(solution, dof_indices);
    // DoFs values in auto-differentiable numbers.
    const std::vector<ADNumber> &dof_values_ad =
        ad_helper.get_sensitive_dof_values();

    std::vector<Tensor<2, dim, ADNumber>> solution_gradient_loc(
        n_q, Tensor<2, dim, ADNumber>());

    fe_values[displacement].get_function_gradients_from_local_dof_values(
        dof_values_ad, solution_gradient_loc);

    std::vector<ADNumber> residual_ad(n_dependent_variables, ADNumber(0.0));

    const std::vector<Point<dim>> &pts = fe_values.get_quadrature_points();

    for (unsigned int q = 0; q < n_q; ++q) {

      // Initialize the orthonormal basis
      initialize_orth_basis(pts[q]);

      // Error checking on the computation of the basis
      if (!std::isfinite(f0[0].val()) || !std::isfinite(n0[0].val())) {
        std::cerr << "Rank " << mpi_rank
                  << " | NaN in Basis at point: " << pts[q]
                  << std::endl;
      }

      // Computing F
      const Tensor<2, dim, ADNumber> F =
          Physics::Elasticity::Kinematics::F(solution_gradient_loc[q]);

      // Computing J
      ADNumber detF = determinant(F);

      // Error checking on J (we need it positive)
      if (detF.val() <= 0.0) {
        std::cerr << "Rank " << mpi_rank
                  << " | Element Inverted! J = " << detF.val() << " at point "
                  << pts[q] << std::endl;
      }

      // Computing C
      const Tensor<2, dim, ADNumber> C = transpose(F) * F;

      //-------------Computing and caching the elements for P--------------------------
      compute_and_cache(i4f, scalar_product(f0, C * f0), intermediate);
      compute_and_cache(i4s, scalar_product(s0, C * s0), intermediate);
      compute_and_cache(i8sf, scalar_product(f0, C * s0), intermediate);

      compute_and_cache(macs, macaulay(i4s - 1), intermediate);
      compute_and_cache(macf, macaulay(i4f - 1), intermediate);

      ADNumber t = exp(bf * intermediate.macf * intermediate.macf);
      compute_and_cache(exp_mac_f_sq, t, intermediate);
      ADNumber t1 = exp(bs * intermediate.macs * intermediate.macs);
      compute_and_cache(exp_mac_s_sq, t1, intermediate);
      ADNumber t2 = exp(bsf * intermediate.i8sf * intermediate.i8sf);
      compute_and_cache(exp_bi8sf_sq, t2, intermediate);

      compute_and_cache(f0f0t, outer_product(f0, f0), intermediate);
      compute_and_cache(s0s0t, outer_product(s0, s0), intermediate);

      compute_and_cache(J, detF, intermediate);

      // We cache the invert explicitly (inspection of tensor.h lines 2850 give
      // ca. 50 memory accesses per inversion of matrix vs. a single cached access
      compute_and_cache(Finv, invert(F), intermediate);
      compute_and_cache(Fmt, transpose(Finv), intermediate);

      // Note: avoid caching J^-2/3, the pade approximation is fast enough and
      // avoids extra memory reads. And no need to cache I1 either since its
      // only useful to compute exp_bi1m3
      ADNumber I1 = pow_m2t(intermediate.J) * trace(C);
      ADNumber t3 = exp(b * (I1 - 3));
      compute_and_cache(exp_bi1m3, t3, intermediate);

      //--------------------------------------------------------------------------------

      compute_P_at_q(F, intermediate);

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        residual_ad[i] +=
            scalar_product(P, fe_values[displacement].gradient(i, q)) *
            fe_values.JxW(q);
      }
    }

    // Neumann & Robin
    if (cell->at_boundary()) {
      for (unsigned int f = 0; f < cell->n_faces(); ++f) {
        int id = cell->face(f)->boundary_id();
        if (cell->face(f)->at_boundary() && (is_neumann(id) || is_robin(id))) {
          fe_face_values.reinit(cell, f);
          if (is_robin(id)) { 
            //------------------------------Robin boundary conditions--------------------------------------------------

            fe_face_values[displacement].get_function_values(
                solution,    // The global solution vector
                val_u_q_surf // Output: The calculated function at all q-points
            );

            Tensor<1, dim, ADNumber> u_q_ad;

            for (unsigned int q = 0; q < n_q_face; ++q) {
              for (unsigned int d = 0; d < dim; ++d)
                u_q_ad[d] = val_u_q_surf[q][d];
              for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                residual_ad[i] +=
                    alfa *
                    scalar_product(fe_face_values[displacement].value(i, q),
                                   val_u_q_surf[q]) *
                    fe_face_values.JxW(q);
              }
            }
            //---------------------------------------------------------------------------------------------------------
          } else { 
           //------------------------------Neumann boundary conditions--------------------------------------------------

            std::vector<Tensor<2, dim, ADNumber>> gradient_bound(
                n_q_face, Tensor<2, dim, ADNumber>());
            fe_face_values[displacement]
                .get_function_gradients_from_local_dof_values(dof_values_ad,
                                                              gradient_bound);
           

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
                Tensor<1, dim, ADNumber> n = fe_face_values.normal_vector(q);
                residual_ad[i] +=
                    p_v *
                    scalar_product(cof * n,
                                   fe_face_values[displacement].value(i, q)) *
                    fe_face_values.JxW(q);
              }
            }
            //-----------------------------------------------------------------------------------------------------------
          }
          
        }
      }
    }

    ad_helper.register_residual_vector(residual_ad);
    ad_helper.compute_residual(cell_rhs);
    cell_rhs *= -1.0;
    ad_helper.compute_linearization(cell_matrix);

    constraints.distribute_local_to_global(cell_matrix, cell_rhs, dof_indices,
                                           jacobian, nr_rhs_f);

  }

  jacobian.compress(VectorOperation::add);
  nr_rhs_f.compress(VectorOperation::add);

  // Dirichlet
  {
    std::map<types::global_dof_index, double> boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    Functions::ZeroFunction<dim> bc_func;
    boundary_functions[PDE_DIRICHLET] = &bc_func;

    VectorTools::interpolate_boundary_values(dof_handler, boundary_functions,
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, jacobian, step_owned,
                                       nr_rhs_f, true);
  }
}

void BarbarottaSolver::initialize_orth_basis(const dealii::Point<dim> &p) {

  constexpr double tol = 1e-8;
  constexpr double eps = 1e-10;
    
  f0.clear();
  n0.clear();
  s0.clear();

  //------------defining t---------------

  const double rho_point =
      std::sqrt((p[0] * p[0] + p[1] * p[1]) / (epi_min * epi_min) +
                (p[2] * p[2]) / (epi_max * epi_max));

  const double rho_endo = endo_min / epi_min;

  double t = (rho_point - rho_endo) / (1.0 - rho_endo);

  t = std::clamp(t, 0.0, 1.0);

  //--------------------------------------

  const double delta_max = std::abs(endo_max - epi_max);
  const double delta_min = std::abs(endo_min - epi_min);

  const double r_s = endo_max + delta_max * t;
  const double r_e = endo_max + delta_min * t;

  const double u = std::acos(p[2] / r_e);

  // Explicit apex handling
  if (std::abs(std::sin(u)) < tol) {
    f0[0] = 0.0;
    f0[1] = 0.0;
    f0[2] = (u > 1.5 ? -1.0 : 1.0);

    n0[0] = 1.0;
    n0[1] = 0.0;
    n0[2] = 0.0;

    s0 = cross_product_3d(n0, f0);
    return;
  }

  const double denom = std::max(std::abs(r_s * std::sin(u)), eps);
  const double v1 = std::asin(p[1] / denom);
  const double v2 = std::acos(p[0] / denom);
  const double v = std::isnan(v1) ? v2 : v1;

  // Fiber rotation law
  const double alpha_deg = 90.0 - 180.0 * t;
  const double alpha = alpha_deg * numbers::PI / 180.0;

  // Computing of dxdu and dxdv
  Tensor<1, dim, double> dxdu;
  Tensor<1, dim, double> dxdv;

  dxdu[0] = r_s * std::cos(u) * std::cos(v);
  dxdu[1] = r_s * std::cos(u) * std::sin(v);
  dxdu[2] = -r_e * std::sin(u);

  dxdv[0] = -r_s * std::sin(u) * std::sin(v);
  dxdv[1] = r_s * std::sin(u) * std::cos(v);
  dxdv[2] = 0.0;

  // Normalization
  dxdu /= dxdu.norm();
  dxdv /= dxdv.norm();

  // Computing the fibers
  const ADNumber sin_a = sin(ADNumber(alpha));
  const ADNumber cos_a = cos(ADNumber(alpha));

  for (unsigned int i = 0; i < dim; ++i)
    f0[i] = ADNumber(dxdu[i]) * sin_a + ADNumber(dxdv[i]) * cos_a;

  // Computing n (pointing outward)
  n0[0] = ADNumber(dxdu[1]) * ADNumber(dxdv[2]) -
          ADNumber(dxdu[2]) * ADNumber(dxdv[1]);
  n0[1] = ADNumber(dxdu[2]) * ADNumber(dxdv[0]) -
          ADNumber(dxdu[0]) * ADNumber(dxdv[2]);
  n0[2] = ADNumber(dxdu[0]) * ADNumber(dxdv[1]) -
          ADNumber(dxdu[1]) * ADNumber(dxdv[0]);

  // Computing s
  s0 = cross_product_3d(n0, f0);


}

void BarbarottaSolver::solve() {
   SolverControl solver_control(20000, 1e-9 * nr_rhs_f.l2_norm());

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

void BarbarottaSolver::solve_newton(const std::string &output_file_name) {
   pcout << "===============================================" << std::endl;

#define MAX_ITER_AMT 180

   const unsigned int n_load_steps = 10; // Number of increments
   const double target_p_v = p_v.val();  // Save the final targets
   const double target_Sn = Sn.val();

   // Outer loop for Load Stepping
   for (unsigned int l_step = 1; l_step <= n_load_steps; ++l_step) {
        double fraction = static_cast<double>(l_step) / n_load_steps;

        // Update the actual parameters used in assemble_system()
        this->p_v = ADNumber(target_p_v * fraction);
        this->Sn = ADNumber(target_Sn * fraction);

        pde_out_c_par(pcout,
                      "\nLOAD STEP " << l_step << "/" << n_load_steps
                                     << " (Load: " << fraction * 100.0 << "%)",
                      YEL_COLOR);

        pde_out_c_par(pcout, "Current p_v: " << p_v << ", Current Sn: " << Sn,
                      YEL_COLOR);

        // --- Inner Newton Loop ---
        const double toll = 1e-6;
        unsigned int n_iter = 0;
        double rhs_norm = toll + 1;

        while (n_iter < MAX_ITER_AMT && rhs_norm > toll) {
          assemble_system();
          rhs_norm = nr_rhs_f.l2_norm();

          pde_out_c_par(pcout,
                        "  Newton iteration " << n_iter << "/" << MAX_ITER_AMT,
                        RED_COLOR);

          pde_out_c_par(pcout,
                        "  Residual norm: " << std::scientific
                                            << std::setprecision(6) << rhs_norm,
                        RED_COLOR);

          if (rhs_norm < toll) {
             break; // Converged for this load step
          } else {

        solve();

        double alpha = (n_iter < 5) ? 0.5 : 1.0;

        solution_owned.add(alpha, step_owned);
        constraints.distribute(solution_owned);
        solution = solution_owned;
          }

          if (std::isnan(rhs_norm)) {
            pde_out_c_par(pcout, "CRITICAL: NaN detected. Aborting.", RED_COLOR);
            return;
          }

          ++n_iter;
        }
        // --- End Inner Newton Loop ---

        pde_out_c_par(pcout, "Load step " << l_step << " converged.",
                      RED_COLOR);
   }

   pde_out_c_par(pcout, "\nSimulation Finished Successfully.", RED_COLOR);
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