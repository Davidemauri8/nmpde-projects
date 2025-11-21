#pragma once
#ifndef __UTILS_VISUALIZE
#define __UTILS_VISUALIZE

#include <iostream>
#include <fstream>
#include <cmath>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/numerics/data_out.h>
 
namespace Mesh {

	/* @brief metafunction for generation of a mesh coloring to represent
	 the domain boundary ids.

	 @param from a path to a mesh .msh file
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

		Triangulation<3>	triangulation;
		GridIn<3>			gridin;

		// Read the mesh file into the helper gridIn class
		std::ifstream input(from);
		gridin.attach_triangulation(triangulation);
		gridin.read_msh(input);

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

} //! namespace Mesh

#endif //! __UTILS_VISUALIZE