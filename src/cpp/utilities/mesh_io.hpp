#pragma once
#ifndef __UTILS_MESH_IO
#define __UTILS_MESH_IO

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <fstream>

#ifndef as_str
// Take the value inside the macro as a literal string
#define as_str(s) #s
#endif

// A set of output macros to debug and show program run data
#ifndef pde_out
#include <ctime>

std::string __tidy_cur_time() {
    static char date[24];
    auto in_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::strftime(date, sizeof(date), "%H:%M:%S", std::gmtime(&in_time_t));
    return date;
}

#ifdef PDE_OUT_VERBOSE
#include <string>
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

// Provide a full specification: [time](file): line.44 |
#define pde_out(s) std::cout << "[" << __tidy_cur_time() << "]" \
    << "(" << (__FILENAME__) << ")" << ":" << __LINE__ << " | " << s << std::endl
#else
#define pde_out(s) std::cout << s << std::endl
#endif //! PDE_OUT_VERBOSE
#endif //! pde_out

namespace Mesh {

    using namespace dealii;

    /* @brief metafunction for loading a mesh into a triangulation

     @param triangulation the mesh into a triangulation
     @param from location of the mesh file
     @tparam D the dimension of the mesh
    */
    template <int D>
    void
        load_mesh_into_tria(
            const std::string& from,
            Triangulation<D>& into
        )
    {
        static_assert(D == 1 || D == 2 || D == 3);
        GridIn<D>			gridin;

        // Read the mesh file into the helper gridIn class
        std::ifstream input(from);
        gridin.attach_triangulation(into);
        if (from.find(".msh") || from.find(".mesh"))
            gridin.read_msh(input);
        else if (from.find(".vtu"))
            gridin.read_vtu(input);
        else gridin.read(input);
        return;
    }

    /* @brief dump a triangulation into a .vtu mesh file

     @param steps the amount of refinement steps
     @param save_into where the mesh is to be saved as a .vtu file
     @tparam D the dimension of the mesh
    */
    template <int D>
    void
        dump_tria(
            const Triangulation<D>& triangulation,
            const std::string& save_into
        )
    {
        static_assert(D == 1 || D == 2 || D == 3);
        std::ofstream mesh_file(save_into);
        GridOut grid_out;
        grid_out.write_vtu(triangulation, mesh_file);
        
        return;
    }

} // !namespace Mesh

#endif //! __UTILS_MESH_IO