
#define PDE_OUT_VERBOSE
#include "utilities/mesh_io.hpp"

#include "solver/superelastic_isotropic.hpp"

void
compute_tensor_from_ref(
	SuperElasticIsotropicSolver::FODTensor& tensor_ref
) {
	tensor_ref = 1.0;
}

int main() {

#define p(s1, s) s
	const std::string save_refined = "generated_meshes/refined_mesh.msh";

	SuperElasticIsotropicSolver seis(
		p("Lagrange Basis Degree", 2),
		p("p_v internal Neumann pressure", 0.5),
		p("Robin condition alpha parameter", 1.0),
		p("Partial derivative of P wrt to F", compute_tensor_from_ref)
	);
	seis.setup(save_refined);
	seis.solve();
#undef p
}