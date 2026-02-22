#include "BoundariesHeartGuccione.hpp"
#include "NeoHookeFibres.hpp"

#include <deal.II/base/function.h>
#include <deal.II/differentiation/ad.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out_faces.h>

#include "../utilities/mesh_io.hpp"
#include <map>

// Defining the method to populate the struct
#define compute_and_cache(name, expression, inter)                             \
  const auto name = expression;                                                \
  inter.name = name;

using ADHelper = Differentiation::AD::ResidualLinearization<
    Differentiation::AD::NumberTypes::sacado_dfad, double>;
using ADNumber = typename ADHelper::ad_type;

void NeoHookeSolver::setup(const std::string &mesh_path) {
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
    pde_out_c_par(pcout, "  Initializing the finite element space", RED_COLOR);

    FE_SimplexP<dim> fe_scalar(r);
    fe = std::make_unique<FESystem<dim>>(fe_scalar, dim);

    pde_out_c_par(pcout, "  Degree = " << r, RED_COLOR);

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

void NeoHookeSolver::compute_P_at_q(const Tensor<2, dim, ADNumber> &F_q,
                                    const pass_cache_data_t &i) {
  P = mu * (F_q - i.Finv) + lambda * (i.J - 1) * i.J * i.Finv;

  // -------- Active Component ---------------

    P += T_a * outer_product(f0, f0);

  // -----------------------------------------

}

void NeoHookeSolver::assemble_system() {
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

      // Computing F
      const Tensor<2, dim, ADNumber> F =
          Physics::Elasticity::Kinematics::F(solution_gradient_loc[q]);
      //-------------Computing and caching--------------------------------------------
      compute_and_cache(Finv, invert(transpose(F), intermediate);
      compute_and_cache(J, determinant(F), intermediate);
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

          std::vector<Tensor<2, dim, ADNumber>> gradient_bound(
              n_q_face, Tensor<2, dim, ADNumber>());
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

void NeoHookeSolver::initialize_orth_basis(const dealii::Point<dim> &p) {

  f0.clear();
  n0.clear();
  s0.clear();

  auto norm = [](const auto &v) {
    auto sum = 0.0;
    for (auto &n : v) {
      sum += n * n;
    }
    return static_cast<decltype(v[0])>(std::sqrt(sum));
  };

  auto normalize = [](auto &v, const auto norm) {
    for (auto &n : v) {
      n /= norm;
    }
  };

  //------defining t---------------

  double rho_point =
      std::sqrt((p[0] * p[0] + p[1] * p[1]) / (epi_min * epi_min) +
                (p[2] * p[2]) / (epi_max * epi_max));

  double rho_endo = endo_min / epi_min;
  double t = (rho_point - rho_endo) / (1.0 - rho_endo);

  t = std::max(0.0, std::min(1.0, t));

  //-------------------------------

  const double endo_epi_max_delta = std::abs(endo_max - epi_max);
  const double endo_epi_min_delta = std::abs(endo_min - epi_min);
  const double r_s = endo_max + endo_epi_max_delta * t;
  const double r_e = endo_max + endo_epi_min_delta * t;
  const double u_rad = std::acos(p[2] / r_e);

  if (std::abs(std::sin(u_rad)) < 1e-8) {

    // At the apex, the Fiber is usually aligned with the longitudinal
    // axis of the ellipsoid (the Z-axis in most heart models).
    f0[0] = 0.0;
    f0[1] = 0.0;
    f0[2] =
        (u_rad > 1.5) ? -1.0 : 1.0; // Direction based on which pole you are at

    // Define a dummy normal and sheet to keep the basis orthonormal
    n0[0] = 1.0;
    n0[1] = 0.0;
    n0[2] = 0.0;
    s0 = cross_product_3d(n0, f0);

  } else {

    const double eps = 1e-10;
    const double r_s_sin_u = r_s * std::sin(u_rad);
    const double safe_denom = (std::abs(r_s_sin_u) < eps) ? eps : r_s_sin_u;
    const double v_rad1 = std::asin(p[1] / (safe_denom));
    const double v_rad2 = std::acos(p[0] / (safe_denom));
    const double v_rad = std::isnan(v_rad1) ? v_rad2 : v_rad1;
    const double alpha_deg = 90.0 - 180.0 * t;
    // const double alpha_deg = 50.0 - 100.0 * t;
    const double alpha_rad = alpha_deg * (M_PI / 180.0);

    // Computing dxdu
    std::vector<double> dxdu = {r_s * std::cos(u_rad) * std::cos(v_rad),
                                r_s * std::cos(u_rad) * std::sin(v_rad),
                                -r_e * std::sin(u_rad)};
    // Computing dxdv
    std::vector<double> dxdv = {
        static_cast<double>(-1) * r_s * std::sin(u_rad) * std::sin(v_rad),
        r_s * std::sin(u_rad) * std::cos(v_rad), static_cast<double>(0)};

    // Normalizing dxdu and dxdv
    normalize(dxdu, norm(dxdu));
    normalize(dxdv, norm(dxdv));

    // Computing f
    for (unsigned i = 0; i < dim; ++i) {
      f0[i] += ADNumber(dxdu[i]) * Sacado::Fad::sin(ADNumber(alpha_rad)) +
               ADNumber(dxdv[i]) * Sacado::Fad::cos(ADNumber(alpha_rad));
    }

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
}

void NeoHookeSolver::solve() {

  SolverControl solver_control(2000, 1e-5 * nr_rhs_f.l2_norm());

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
                "  " << solver_control.last_step() << " GMRES iterations",
                RED_COLOR);
}

void NeoHookeSolver::solve_newton(const std::string &output_file_name) {
  pcout << "===============================================" << std::endl;
#define MAX_ITER_AMT 80

  const double toll = 1e-6;
  unsigned int n_iter = 0;
  double rhs_norm = toll + 1;

  while (n_iter < MAX_ITER_AMT && rhs_norm > toll) {
    assemble_system();
    rhs_norm = nr_rhs_f.l2_norm();

    pde_out_c_par(pcout, "Newton iteration " << n_iter << "/" << MAX_ITER_AMT,
                  RED_COLOR);

    pde_out_c_par(pcout,
                  "The norm of the residual is: "
                      << std::scientific << std::setprecision(6) << rhs_norm,
                  RED_COLOR);

    if (rhs_norm < toll) {
      n_iter = MAX_ITER_AMT;
    } else {
      solve();

      solution_owned += step_owned;
      constraints.distribute(solution_owned);
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
