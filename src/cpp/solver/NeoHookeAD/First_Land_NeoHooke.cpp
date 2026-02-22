#include "NeoHookeFirstLand.hpp"

// Main function.
int main(int argc, char *argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int r = 1;

  NeoHookeSolver problem(r);

  problem.setup();
  problem.solve_newton("NeoHooke_heart.vtk");

  return 0;
}