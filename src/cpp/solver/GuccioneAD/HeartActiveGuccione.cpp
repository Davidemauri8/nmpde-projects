#include "BoundariesHeartGuccione.hpp"
#include "HeartActiveGuccione.hpp"

#include <deal.II/base/function.h>
#include <deal.II/differentiation/ad.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out_faces.h>

#include <algorithm>
#include "../utilities/mesh_io.hpp"
#include <map>

// Defining the method to populate the struct
#define compute_and_cache(name, expression, inter)                             \
  const auto name = expression;                                                \
  inter.name = name;

using ADHelper = Differentiation::AD::ResidualLinearization<
    Differentiation::AD::NumberTypes::sacado_dfad, double>;
using ADNumber = typename ADHelper::ad_type;

void GuccioneSolver::setup(const std::string &mesh_path) {
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

    pde_out_c_par(pcout, "  Degree                      = " << fe->degree,
                  RED_COLOR);
    pde_out_c_par(pcout,
                  "  DoFs per cell               = " << fe->dofs_per_cell,
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

void GuccioneSolver::compute_P_at_q(const Tensor<2, dim, ADNumber> &F_q,
                                    const pass_cache_data_t &i) {

  //--------------------------computing e^Q---------------------------------
  ADNumber normal = bn0n0 * (i.En0n0 * i.En0n0);
  ADNumber ff = bf0f0 * (i.Ef0f0 * i.Ef0f0);
  ADNumber ss = bs0s0 * (i.Es0s0 * i.Es0s0);
  ADNumber mutual = bf0s0 * (i.Ef0s0 * i.Ef0s0 + i.Es0f0 * i.Es0f0) +
                    bf0n0 * (i.Ef0n0 * i.Ef0n0 + i.En0f0 * i.En0f0) +
                    bs0n0 * (i.Es0n0 * i.Es0n0 + i.En0s0 * i.En0s0);

  ADNumber Q = normal + ff + ss + mutual;

  if (!std::isfinite(Q.val())) {
    pcout << "Q is not finite!" << std::endl;
  }

  if (Q.val() > 50) {
    std::cout << "Q too large: " << Q.val() << std::endl;
  }


  ADNumber expQ = exp(Q);
  //------------------------------------------------------------------------

  //--------------------------computing dQdE-------------------------------

  Tensor<2, dim, ADNumber> dQdE;
  dQdE = 2.0 * bf0f0 * i.Ef0f0 * i.f0f0 + 2.0 * bs0s0 * i.Es0s0 * i.s0s0 +
         2.0 * bn0n0 * i.En0n0 * i.n0n0 +
         2.0 * bf0s0 * i.Ef0s0 * (i.f0s0 + i.s0f0) +
         2.0 * bf0n0 * i.Ef0n0 * (i.f0n0 + i.n0f0) +
         2.0 * bs0n0 * i.Es0n0 * (i.s0n0 + i.n0s0);
  //------------------------------------------------------------------------

  Tensor<2, dim, ADNumber> S_pass = 0.5 * C * expQ * dQdE;

  //--------------------------active component------------------------------

  Tensor < 2, dim, ADNumber > S_act= T_a * outer_product(f0, f0);

  //------------------------------------------------------------------------

  //--------------------------volumetric component--------------------------
  Tensor<2, dim, ADNumber> FinvT = invert(transpose(F_q));
  ADNumber J = determinant(F_q);
  Tensor<2, dim, ADNumber> S_vol = bulk * (J - 1.0) * J * FinvT;
  //------------------------------------------------------------------------

  Tensor<2, dim, ADNumber> S = S_act + S_pass;

  P = F_q*S + S_vol;
}

void GuccioneSolver::assemble_system() {
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();
  const unsigned int n_q_face = quadrature_face->size();

  FEValues<dim> fe_values(*fe, *quadrature,
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

      // Computing F and E
      const Tensor<2, dim, ADNumber> F =
          Physics::Elasticity::Kinematics::F(solution_gradient_loc[q]);
      const ADNumber J = determinant(F);
      // Debugging print
      if (q == 0 && cell->is_locally_owned() &&
          (J.val() < 0.9 || J.val() > 1.2))
        std::cout << "J=" << J.val() << std::endl;

      const Tensor<2, dim, ADNumber> E = Physics::Elasticity::Kinematics::E(F);

      //-------------Computing and caching the elements for computing
      //e^Q---------------
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

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        residual_ad[i] +=
            (scalar_product(P, fe_values[displacement].gradient(i, q))) *
            fe_values.JxW(q);
      }
    }

    // Neumann
    if (cell->at_boundary()) {
      for (unsigned int f = 0; f < cell->n_faces(); ++f) {
        int id = cell->face(f)->boundary_id();
        if (cell->face(f)->at_boundary() && is_neumann(id)) {
          fe_face_values.reinit(cell, f);

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

void GuccioneSolver::initialize_orth_basis(const dealii::Point<dim> &p) {
  // Formulation following the procedure described here: 
  // https://pmc.ncbi.nlm.nih.gov/articles/PMC4707707/pdf/rspa20150641.pdf

  constexpr double tol = 1e-8;
  constexpr double eps = 1e-10;

  f0.clear();
  n0.clear();
  s0.clear();

  //------------defining t---------------

  auto rs = [&](double t) { return endo_min + (epi_min - endo_min) * t; };
  auto re = [&](double t) { return endo_max + (epi_max - endo_max) * t; };

  auto g = [&](double t) {
    const double rs_t = rs(t), rl_t = re(t);
    return (p[0] * p[0] + p[1] * p[1]) / (rs_t * rs_t) +
           (p[2] * p[2]) / (rl_t * rl_t) - 1.0;
  };

  // Bisection on [0,1]
  double a = 0.0, b = 1.0;
  double fa = g(a), fb = g(b);
  double t = 0;
  // if point is slightly outside due to numerics, clamp
  if (fa * fb > 0.0) {
    // fallback: closest end
    t = (std::abs(fa) < std::abs(fb)) ? a : b;
    // use t
  } else {
    for (int it = 0; it < 50; ++it) {
      double m = 0.5 * (a + b);
      double fm = g(m);
      if (fa * fm <= 0.0) {
        b = m;
        fb = fm;
      } else {
        a = m;
        fa = fm;
      }
    }
    t = 0.5 * (a + b);
  }
  //--------------------------------------

  const double r_s = rs(t);
  const double r_e = re(t);

  // Use atan2 for u (more robust than acos)
  const double sin_u = std::sqrt(p[0] * p[0] + p[1] * p[1]) / r_s;
  const double cos_u = p[2] / r_e;
  const double u = std::atan2(sin_u, cos_u); // gives u in [0,pi]

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

  const double v = std::atan2(p[1], p[0]);

  // Fiber rotation law
  const double alpha_deg = (90.0 - 180.0 * t)*-1;
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

  // After f0 computed:
  const ADNumber fnorm = std::sqrt(f0 * f0);
  if (fnorm > 0)
    f0 /= fnorm;

  // n0 from cross(dxdu,dxdv)
  const ADNumber nnorm = std::sqrt(n0 * n0);
  if (nnorm > 0)
    n0 /= nnorm;

  // s0 = n0 x f0, then normalize
  s0 = cross_product_3d(n0, f0);
  const ADNumber snorm = std::sqrt(s0 * s0);
  if (snorm > 0)
    s0 /= snorm;

  // (optional) re-orthogonalize f0 to remove drift:
  f0 = cross_product_3d(s0, n0);
  const ADNumber fnorm2 = std::sqrt(f0 * f0);
  if (fnorm2 > 0)
    f0 /= fnorm2;


}

void GuccioneSolver::solve() {
  SolverControl solver_control(7000, 1e-5 * nr_rhs_f.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);
  //SolverBicgstab<TrilinosWrappers::MPI::Vector> solver(solver_control);
  
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
  

  //TrilinosWrappers::PreconditionSSOR preconditioner;
  //preconditioner.initialize(jacobian);

  solver.solve(jacobian, step_owned, nr_rhs_f, preconditioner);

  pde_out_c_par(pcout,
                "  " << solver_control.last_step() << " GMRES iterations",
                RED_COLOR);
}

void GuccioneSolver::solve_newton(const std::string &output_file_name) {
  pcout << "===============================================" << std::endl;

  #define MAX_ITER_AMT 1000

  const unsigned int n_load_steps = 15; // Number of increments
  const double target_p_v = p_v.val();  // Save the final targets
  const double target_Ta = T_a.val();


  // Outer loop for Load Stepping
  for (unsigned int l_step = 1; l_step <= n_load_steps; ++l_step) {
    double fraction = static_cast<double>(l_step) / n_load_steps;

    // Update the actual parameters used in assemble_system()
    this->p_v = ADNumber(target_p_v * fraction);
    this->T_a = ADNumber(target_Ta * fraction);

    pde_out_c_par(pcout,
                  "\nLOAD STEP " << l_step << "/" << n_load_steps
                                 << " (Load: " << fraction * 100.0 << "%)",
                  YEL_COLOR);

    pde_out_c_par(pcout, "Current p_v: " << p_v << ", Current T_a: " << T_a,
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

        double alpha = 1.0;

        if (n_iter < 35) {
          alpha = 0.1;
        } else if (n_iter < 60) {
          alpha = 0.2;
        } else if (n_iter < 80) {
          alpha = 0.3;
        } else if (n_iter < 100) {
          alpha = 0.4;
        } else if (n_iter < 130) {
          alpha = 0.5;
        } else if (n_iter < 170) {
          alpha = 0.6;
        } else if (n_iter < 250) {
          alpha = 0.7;
        } else {
          alpha = 0.8;
        }

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

    pde_out_c_par(pcout, "Load step " << l_step << " converged.", RED_COLOR);
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

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record("./", output_file_name, 0,
                                      MPI_COMM_WORLD);

  pde_out_c_par(pcout, "Result saved into file " << output_file_name,
                GRN_COLOR);
  //-------------------------------------------------------------------------
}

