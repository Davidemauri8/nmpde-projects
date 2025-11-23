
#define PDE_OUT_VERBOSE
#include "utilities/mesh_io.hpp"

#include "solver/superelastic_isotropic.hpp"

int main() {

	const std::string save_refined = "generated_meshes/refined_mesh.msh";

	SuperElasticIsotropicSolver seis(
		2, 0.5, true);
	seis.setup(save_refined);
}