#pragma once
#ifndef __UTILS_VISUALIZE
#define __UTILS_VISUALIZE

#include <iostream>
#include <fstream>
#include <cmath>
#include <functional>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/numerics/data_out.h>
 
#include "../solver/mesh_geometry.hpp"
#include "../utilities/mesh_io.hpp"

namespace UtilsMesh {

	/* @brief metafunction for generation of a mesh coloring to represent
	 the domain boundary ids.

	 @param from a path to a mesh .msh file or .vtu file
	 @param save_into where the mesh coloring is to be saved as a .vtu file
	 @tparam D the dimension of the mesh
	*/
	template <int D>
	void
		boundary_view_mapping(
			const std::string& from,
			const std::string& save_into
		) {
		using namespace dealii;

		// Load the mesh into the triangulation
		Triangulation<3>	triangulation;
		UtilsMesh::load_mesh_into_tria(from, triangulation);

		DataOut<3> data_out;
		data_out.attach_triangulation(triangulation);

		// Attach boundary IDs into a dummy vector
		// NOTE: In a not-so-fine mesh, this may fail as a single voxel (cell)
		// can be attached to multiple boundary surfaces, hence the coloring is
		// not well defined in general!
		// Still this can be used to determine whether the boundary surfaces
		// do exist.

		Vector<float> boundary_indicator(triangulation.n_active_cells());
		boundary_indicator = 0.0;
		for (const auto& cell : triangulation.active_cell_iterators()) {
			for (unsigned int f = 0; f < GeometryInfo<D>::faces_per_cell; ++f) {
				// Select faces of cells which are at a boundary and have non-zero boundary id
				if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() != 0)
					boundary_indicator(cell->index()) = cell->face(f)->boundary_id();
			}
		}

		data_out.add_data_vector(boundary_indicator, "boundary_id");

		std::ofstream output(save_into);
		data_out.build_patches();
		data_out.write_vtu(output);

		return;
	}

	template <int D>
	void
		view_cartesian_coords(
			const Triangulation<3>& triangulation,
			std::unique_ptr<FiniteElement<D>>& fe,
			std::unique_ptr<Quadrature<D>>& quadrature,
			const std::string save_into
		) {
		using namespace dealii;

		// Load the mesh into the triangulation

		DataOut<D> data_out;
		data_out.attach_triangulation(triangulation);

		// FEValues instance. This object allows to compute basis functions, their
		// derivatives, the reference-to-current element mapping and its
		// derivatives on all quadrature points of all elements.
		FEValues<D> fe_values(*fe, *quadrature,
			update_values | update_gradients | update_quadrature_points |
			update_JxW_values);

		Vector<float> x_vec(triangulation.n_active_cells());
		Vector<float> y_vec(triangulation.n_active_cells());
		Vector<float> z_vec(triangulation.n_active_cells());

		x_vec = 0.0; y_vec = 0.0; z_vec = 0.0;
		for (const auto& cell : triangulation.active_cell_iterators()) {
			const auto ci = cell->index();
			fe_values.reinit(cell);

			const auto point = cell->center();

			x_vec[ci] = (float)point[0];
			y_vec[ci] = (float)point[1];
			z_vec[ci] = (float)point[2];
		}

		data_out.add_data_vector(x_vec, "x");
		data_out.add_data_vector(y_vec, "y");
		data_out.add_data_vector(z_vec, "z");

		std::ofstream output(save_into);
		data_out.build_patches();
		data_out.write_vtu(output);

		return;
	}

	template <int D = 3>
	void visualize_grain_fibers(
		std::function<void(const dealii::Point<3, double>&, std::vector<dealii::Tensor<1, 3>>&)> f,
		const Triangulation<D>& triangulation,
		std::unique_ptr<FiniteElement<D>>& fe,
		std::unique_ptr<Quadrature<D>>& quadrature,
		const std::string save_into
	) {
		using namespace dealii;

		// Load the mesh into the triangulation

		DataOut<D> data_out;
		data_out.attach_triangulation(triangulation);

		// FEValues instance. This object allows to compute basis functions, their
		// derivatives, the reference-to-current element mapping and its
		// derivatives on all quadrature points of all elements.
		FEValues<D> fe_values(*fe, *quadrature,
			update_values | update_gradients | update_quadrature_points |
			update_JxW_values);

		// Its inelegant but required because dealii stores just a reference to these
		// vectors when data_out.add_data_vector is called, so they need to be
		// different memory locations until .build_patches() is called
		Vector<double> ux(triangulation.n_active_cells());
		Vector<double> uy(triangulation.n_active_cells());
		Vector<double> uz(triangulation.n_active_cells());
		Vector<double> vx(triangulation.n_active_cells());
		Vector<double> vy(triangulation.n_active_cells());
		Vector<double> vz(triangulation.n_active_cells());
		Vector<double> wx(triangulation.n_active_cells());
		Vector<double> wy(triangulation.n_active_cells());
		Vector<double> wz(triangulation.n_active_cells());

		std::vector<Tensor<1, D>> orth_basis(D);

		for (const auto& cell : triangulation.active_cell_iterators()) {
			const auto ci = cell->index();
			fe_values.reinit(cell);
			const auto point = cell->center();

			f(point, orth_basis);
			const auto& u = orth_basis[0];
			const auto& v = orth_basis[1];
			const auto& w = orth_basis[2];

			ux[ci] = u[0]; uy[ci] = u[1]; uz[ci] = u[2];
			vx[ci] = v[0]; vy[ci] = v[1]; vz[ci] = v[2];
			wx[ci] = w[0]; wy[ci] = w[1]; wz[ci] = w[2];

		}

		const std::vector<Vector<double>> vecs = { ux, uy, uz, vx, vy, vz, wx, wy, wz };
		const std::vector<std::string> names = { "ux", "uy", "uz", "vx", "vy", "vz", "wx", "wy", "wz" };
		for (int i = 0; i < 9; ++i)
			data_out.add_data_vector(vecs[i], names[i]);

		std::ofstream output(save_into);
		data_out.build_patches();
		data_out.write_vtu(output);
	}


	template <int D = 3>
	void visualize_wall_depth(
		const Triangulation<D>& triangulation,
		const std::string save_into
	) {
		using namespace dealii;

		// Load the mesh into the triangulation

		DataOut<D> data_out;
		data_out.attach_triangulation(triangulation);

		// FEValues instance. This object allows to compute basis functions, their
		// derivatives, the reference-to-current element mapping and its
		// derivatives on all quadrature points of all elements.

		Vector<double> wall_depth(triangulation.n_active_cells());

		for (const auto& cell : triangulation.active_cell_iterators()) {
			const auto ci = cell->index();

			const auto p = cell->center();

			const double x = p[0];
			const double y = p[1];
			const double z = p[2];

			const double csquared = MESH_ELLIPSOID_Z_DEFORMATION * MESH_ELLIPSOID_Z_DEFORMATION;

			const double r = std::sqrt(x * x + y * y + z * z / (csquared* csquared));

			const double r_over_width = 
				(r - MESH_ELLIPSOID_SMALL_RADIUS) / (MESH_ELLIPSOID_LARGE_RADIUS - MESH_ELLIPSOID_SMALL_RADIUS);

			wall_depth[ci] = r_over_width;
		}

		data_out.add_data_vector(wall_depth, "wall_depth");

		std::ofstream output(save_into);
		data_out.build_patches();
		data_out.write_vtu(output);


	}

} //! namespace UtilsMesh

#endif //! __UTILS_VISUALIZE