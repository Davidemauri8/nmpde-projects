
#define PDE_OUT_VERBOSE
#include "utilities/mesh_io.hpp"
#include "utilities/visualize.hpp"
#include "solver/superelastic_isotropic.hpp"

#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>

#include <ctime>



int main() {

#define p(s1, s) s
	const std::string save_refined = "generated_meshes/clean_heart_cup.msh";
	const std::string save_into = "clean_boundary.vtu";

	SuperElasticIsotropicSolver seis(
		p("Lagrange Basis Degree", 2),
		p("p_v internal Neumann pressure", 0.15),
		p("Robin condition alpha parameter", 0.25),
		p("Mu value", 0.15),
		p("Bulk penalty", 0.1),
		p("Anisotropic Af", 0.5),
		p("Anisotropic As", 1)
		// p("Partial derivative of P wrt to F", compute_tensor_from_ref)
	);
	seis.setup(save_refined);
	UtilsMesh::boundary_view_mapping<3>(save_refined, save_into);
	seis.solve();
#undef p
}