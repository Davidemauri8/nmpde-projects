#pragma once
#ifndef __SOLVER_BOUNDARIES

/* =========== PREVIOUS MESH VALUES ===========
#define PDE_ROBIN		2
#define PDE_DIRICHLET	3
#define PDE_NEUMANN		4
*/
#define PDE_ROBIN		3
#define PDE_DIRICHLET	1
#define PDE_NEUMANN		2


#define is_robin(id) (id == PDE_ROBIN)
#define is_neumann(id) (id == PDE_NEUMANN)
#define is_dirichlet(id) (id == PDE_DIRICHLET)

#define is_not_a_boundary(id) (id != PDE_ROBIN && id != PDE_NEUMANN && id != PDE_DIRICHLET)
#define is_a_boundary(id) (!is_not_a_boundary(id))
#endif //! __SOLVER_BOUNDARIES