#pragma once
#ifndef __SOLVER_BOUNDARIES

#define PDE_DIRICHLET 0
#define PDE_NEUMANN 4

#define is_neumann(id) (id == PDE_NEUMANN)
#define is_dirichlet(id) (id == PDE_DIRICHLET)


#define is_not_a_boundary(id) (id != PDE_NEUMANN && id != PDE_DIRICHLET)
#define is_a_boundary(id) (!is_not_a_boundary(id))
#endif //! __SOLVER_BOUNDARIES