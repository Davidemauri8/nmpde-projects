#include "HeartActiveGuccione.hpp"

// Main function.
int main(int argc, char *argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int r = 2;
  const std::string mesh_path =
      "generated_meshes/coarse_heart_land.msh";
  //const std::string mesh_path =
  //    "generated_meshes/fine_land.msh";

  GuccioneSolver problem(r);

  problem.setup(mesh_path);
  problem.solve_newton("Guccione_active_heart.vtk");

  return 0;
}