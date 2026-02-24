// LV parameters [m].
h = 0.0005; // Mesh size

// Surface parameters
r_s_endo = 0.007;  // m
r_l_endo = 0.017;  // m
r_s_epi  = 0.010;  // m
r_l_epi  = 0.020;  // m

// Paper ranges: u in [-Pi, -acos(5/r_l)]
u_base_endo = -Acos(5 / 17.0);
u_base_epi  = -Acos(5 / 20.0);

//////////////////////////////////////////////////////////
// Function for point in Ellipsoid Coordinates (paper)  //
//////////////////////////////////////////////////////////
Function EllipsoidPoint
  Point(id) = { r_s * Sin(u) * Cos(v),
                r_s * Sin(u) * Sin(v),
                r_l * Cos(u),
                h };
Return

//////////////////////////////////////////////////////////

// Center point (needed by Ellipse)
center = newp;
Point(center) = {0.0, 0.0, 0.0, h};

v = 0.0;

// Endo
r_s = r_s_endo; r_l = r_l_endo;

u = -Pi;
apex_endo = newp; id = apex_endo; Call EllipsoidPoint;

u = u_base_endo;
base_endo = newp; id = base_endo; Call EllipsoidPoint;

// Epi
r_s = r_s_epi; r_l = r_l_epi;

u = -Pi;
apex_epi = newp; id = apex_epi; Call EllipsoidPoint;

u = u_base_epi;
base_epi = newp; id = base_epi; Call EllipsoidPoint;

/////////////////////////// Left ventricle ///////////////////////////

Ellipse(1) = {base_endo, center, apex_endo, apex_endo};
Ellipse(2) = {base_epi,  center, apex_epi,  apex_epi};
Line(3) = {apex_epi, apex_endo};
Line(4) = {base_endo, base_epi};
Line Loop(1) = {1, -3, -2, -4};

Plane Surface(1) = {1};

// Revolve 4x to make full 360°
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} { Surface{1}; }
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} { Surface{21}; }
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} { Surface{38}; }
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} { Surface{55}; }


Physical Volume("Myocardium")   = {1, 2, 3, 4};
Physical Surface("Basal plane") = {37, 54, 71, 20};
Physical Surface("Endocardium") = {29, 12, 46, 63};
Physical Surface("Epicardium")  = {16, 33, 50, 67};

Mesh 3;