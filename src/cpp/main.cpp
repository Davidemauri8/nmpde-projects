#include "utilities/visualize.hpp"

int main() {
	using namespace Mesh;

	const std::string ref_mesh = "generated_meshes/heart_cup.msh";
	const std::string save_mesh = "generated_meshes/vertex_vis.vtu";

	boundary_view_mapping<3>(
		ref_mesh, save_mesh
	);
	cout << "Saved the reference mesh " << ref_mesh << " into " << save_mesh << std::endl; 
}