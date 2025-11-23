#pragma once
#ifndef __SOLVER_BOUNDARIES

#define PDE_ROBIN		2
#define PDE_DIRICHLET	3
#define PDE_NEUMANN		4

#define is_robin(id) (id == PDE_ROBIN)
#define is_neumann(id) (id == PDE_NEUMANN)
#define is_dirichlet(id) (id == PDE_DIRICHLET)

#define not_a_boundary(id) (id != PDE_ROBIN && id != PDE_NEUMANN && id != PDE_DIRICHLET)

#endif //! __SOLVER_BOUNDARIES