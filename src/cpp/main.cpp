
#define PDE_OUT_VERBOSE
#include "utilities/mesh_io.hpp"
#include "utilities/visualize.hpp"
#include "solver/complete_solver.hpp"

#include <ctime>
#include "solver/derivatives.hpp"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <fenv.h>
#include <string>

#include <utility>

void
parse_params(int argc, char* argv[], std::vector<double>& values) {
	for (int i = 1; i < argc; ++i) {
		values[i-1] = std::stod(std::string(argv[i]));
	}
}

using namespace dealii;

int main(int argc, char* argv[]) {

	// Parameters for the model are taken in input from the line command
	// example activation: 
	//  mpirun -n 4 ./main 2 0.1 0.04 0.3 4.5 4.19 6.5 2.5 10.4 0.130 15.255 2.0 0.5 9.0

	Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

	// ENABLE ALL EXCEPTIONS FOR NAN GENERATION TO SEE WHERE THINGS GO ROGUE
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

	std::vector<double> vals(100);
	parse_params(argc, argv, vals);

	// Show the name, a reference value and the real value taken as line argument
#define p(s1, def, s) s

	const std::string save_refined = "generated_meshes/clean_heart_cup.msh";
	const std::string save_into = "clean_boundary.vtu";

	OrthotropicSolver seis(
		p("Lagrange Basis Degree", 2, (int) vals[0]),
		p("p_v internal Neumann pressure", 2, vals[1]),
		p("Robin condition alpha parameter", 1, vals[2]),
		p("a value", 0.2, vals[3]),
		p("b value", 4, vals[4]),
		p("af value", 0.5, vals[5]),
		p("bf value", 0.5, vals[6]),
		p("as value", 0.5, vals[7]),
		p("bs value", 0.5, vals[8]),
		p("asf value", 0.5, vals[9]),
		p("bsf value", 0.5, vals[10]),
		p("Sn value", 0.5, vals[11]),
		p("beta value", 0.5, vals[12]),
		p("Bulk penalty", 7, vals[13])
	);

	seis.setup(save_refined);
	// UtilsMesh::boundary_view_mapping<3>(save_refined, save_into);
	seis.solve("ortho_z_19_ACTIVE_CONTRACTION_2.vtk");
	
#undef p
}