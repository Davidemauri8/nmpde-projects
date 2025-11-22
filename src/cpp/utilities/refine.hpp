#pragma once
#ifndef __UTILS_REFINE
#define __UTILS_REFINE

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <fstream>

namespace Mesh {

    using namespace dealii;


    /* @brief metafunction for refinement of a mesh up to n steps of refinement.

     @param triangulation the mesh into a triangulation
     @param steps the amount of refinement steps
     @param save_into where the mesh is to be saved as a .vtu file
     @tparam D the dimension of the mesh
    */
    template <int D>
    void
    refine_dmesh_nsteps(
        Triangulation<D>& triangulation,
        const unsigned int steps,
        const std::string& save_into
    )
    {
        triangulation.refine_global(steps);
         
        std::ofstream mesh_file(save_into);
        GridOut grid_out;
        // Set the first flag (save boundary ids) to true, the second 
        // one to false (saves "boundary lines", somehow causes a segmentation
        // fault...)
        static const auto flags = GridOutFlags::Msh(true, false);
        grid_out.set_flags(flags);

        grid_out.write_msh(triangulation, mesh_file);

    }

} // !namespace Mesh

#endif //! __UTILS_REFINE