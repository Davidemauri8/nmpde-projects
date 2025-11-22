
#define PDE_OUT_VERBOSE
#include "utilities/mesh_io.hpp"
#include "utilities/refine.hpp"
#include "utilities/visualize.hpp"

int main() {
	using namespace Mesh;

	const std::string ref_mesh = "generated_meshes/heart_cup.msh";
	const std::string save_mesh = "generated_meshes/vertex_vis.vtu";
	const std::string save_refined = "generated_meshes/refined_mesh.msh";

	
	Triangulation<3> triangulation;
	pde_out("Loading the mesh");

	Mesh::load_mesh_into_tria(ref_mesh, triangulation);

	pde_out("Refining the mesh");
	 Mesh::refine_dmesh_nsteps(
	 	triangulation, 2, save_refined
	);
	
	pde_out("Visualizing the boundaries");
	Mesh::boundary_view_mapping<3>(
		save_refined, save_mesh
	);
	pde_out("Saved the refined mesh of " << ref_mesh << " into " << save_mesh); 
}