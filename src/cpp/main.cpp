
#define PDE_OUT_VERBOSE
#include "utilities/mesh_io.hpp"
#include "utilities/visualize.hpp"
#include "solver/orthotropic_solver.hpp"

#include <ctime>
#include "solver/derivatives.hpp"

using namespace dealii;

int main() {

#define p(s1, s) s
	const std::string save_refined = "generated_meshes/clean_heart_cup.msh";
	const std::string save_into = "clean_boundary.vtu";

	SuperElasticOrthotropicSolver seis(
		p("Lagrange Basis Degree", 2),
		p("p_v internal Neumann pressure", 0.15),
		p("Robin condition alpha parameter", 0.25),
		p("Mu value", 0.15),
		p("Bulk penalty", 0.1),
		p("Anisotropic Af", 0.5),
		p("Anisotropic As", 1),
		p("Cross term Afs", 0.2)
	);
	seis.setup(save_refined);
	// UtilsMesh::boundary_view_mapping<3>(save_refined, save_into);
	seis.solve();
	
#undef p
}