#include "Barbarotta.hpp"

// Main function.
int main(int argc, char *argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int r = 2;
  // fine mesh
  //const std::string mesh_path = "generated_meshes/heart-0005.msh";
  // "medium" mesh
  //const std::string mesh_path = "generated_meshes/heart-0025.msh";
  // coarse mesh
  //const std::string mesh_path = "generated_meshes/heart-005.msh";
  //const std::string mesh_path =
  //  "generated_meshes/reversed_coarse_heart.msh";
  const std::string mesh_path = "generated_meshes/ms.msh";

  BarbarottaSolver problem(r);

  problem.setup(mesh_path);
  problem.solve_newton("Barbarotta.vtk");

  return 0;
}