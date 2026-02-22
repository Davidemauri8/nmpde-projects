#include "NeoHookeHeart.hpp"

// Main function.
int main(int argc, char *argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int r = 2;
  // fine mesh
  // const std::string mesh_path = "generated_meshes/heart-0005.msh";
  // "medium" mesh
  const std::string mesh_path = "generated_meshes/heart-0025.msh";
  // coarse mesh
  // const std::string mesh_path = "generated_meshes/heart-005.msh";

  NeoHookeSolver problem(r);

  problem.setup(mesh_path);
  problem.solve_newton("NeoHooke_heart.vtk");

  return 0;
}