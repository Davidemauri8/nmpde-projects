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

#define RED_COLOR "\x1b[0;31m"
#define GRN_COLOR "\x1b[0;32m"
#define YEL_COLOR "\x1b[0;33m"
#define BLU_COLOR "\x1b[0;36m"
#define COLOR_RESET "\x1b[0m"

#define FINE_LVL 1
#define FINER_LVL 2
#define FINEST_LVL 3

// A set of output macros to debug and show program run data
#undef pde_out
#ifndef pde_out

#ifdef PDE_OUT_VERBOSE
#pragma message "INFO (NOT an error): PDE_OUT_VERBOSE was defined, so verbose output macros will be generated for this unit."
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

std::string
__tidy_cur_time();

// Provide a full specification: [time](file): line.44 |
#define _pde_str_shape(s) "[" << __tidy_cur_time() << "]" \
    << "(" << (__FILENAME__) << ")" << ":" << __LINE__ << " | " << s << std::endl

#else
#define _pde_str_shape(s) s << std::endl
#endif //! PDE_OUT_VERBOSE

#define pde_out(s) std::cout << _pde_str_shape(s)
#define pde_out_c(s, c) std::cout << c << _pde_str_shape(s) << COLOR_RESET;
#define pde_out_i(s, i) std::cout << std::string(3 * i, '-') << _pde_str_shape(s)
#define pde_out_c_i(s, c, i) std::cout << c  << " \'" << std::string(3 * i, '-')  << "> " << _pde_str_shape(s) << COLOR_RESET;

#endif //! pde_out


namespace UtilsMesh {

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

} // !namespace UtilsMesh

#endif //! __UTILS_MESH_IO