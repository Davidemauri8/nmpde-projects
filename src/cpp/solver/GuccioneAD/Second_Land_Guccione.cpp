#include "HeartGuccioneSolver.hpp"

// Main function.
int main(int argc, char *argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int r = 1;
  // fine mesh
  // const std::string mesh_path = "generated_meshes/heart-0005.msh";
  // "medium" mesh
  //const std::string mesh_path = "generated_meshes/heart-0025.msh";
  // coarse mesh
  //const std::string mesh_path = "generated_meshes/heart-005.msh";

  //const std::string mesh_path = "generated_meshes/land_twisted.msh";

  const std::string mesh_path = "generated_meshes/ms.msh";

  //const std::string mesh_path =
  //   "generated_meshes/coarse_heart_land.msh";
  //const std::string mesh_path =
  //    "generated_meshes/finest_land.msh";

  GuccioneSolver problem(r);

  problem.setup(mesh_path);
  problem.solve_newton("Guccione_heart.vtk");

  return 0;
}