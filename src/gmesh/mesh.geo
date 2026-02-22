//////////////////////////////////////////////////////////////
//// This work is licensed under a Creative Commons       ////
//// Attribution-NonCommercial 4.0 International License. ////
//////////////////////////////////////////////////////////////

// LV parameters [m].

h = 0.0005; // Mesh size

// Surface parameters
r_s_endo = 0.007;  // m, endocardial surface
r_l_endo = 0.017;  // m, endocardial surface
r_s_epi = 0.010;   // m, epicardial surface
r_l_epi = 0.020;   // m, epicardial surface

mu_base_endo = Acos(5 / 17.0);  
mu_base_epi = Acos(5 / 20.0);   

//////////////////////////////////////////////////////////
// Function for point in Ellipsoid Coordinates          //
//////////////////////////////////////////////////////////

Function EllipsoidPoint
        Point(id) = { r_s * Sin(mu) * Cos(theta),
                      r_s * Sin(mu) * Sin(theta),
                      r_l * Cos(mu),
                      h };
Return

//////////////////////////////////////////////////////////

// Center, point (tag 1)
center = newp;
Point(center) = {0.0, 0.0, 0.0};

// Apex_L Endo, point (tag 2)
theta = 0.0;
r_s = r_s_endo;
r_l = r_l_endo;
mu = 0.0;
apex_endo = newp;
id = apex_endo;
Call EllipsoidPoint;

// Base_L Endo, point (tag 3)
mu = mu_base_endo;
base_endo = newp;
id = base_endo;
Call EllipsoidPoint;

// Apex_L Epi, point (tag 4)
r_s = r_s_epi;
r_l = r_l_epi;
mu = 0.0;
apex_epi = newp;
id = apex_epi;
Call EllipsoidPoint;

// Base_L Epi, point (tag 5)
mu = mu_base_epi;
base_epi = newp;
id = base_epi;
Call EllipsoidPoint;

/////////////////////////// Left ventricle ///////////////////////////

Ellipse(1) = {base_endo, center, apex_endo, apex_endo};
Ellipse(2) = {base_epi, center, apex_epi, apex_epi};
Line(3) = {apex_epi, apex_endo};
Line(4) = {base_endo, base_epi};
Line Loop(1) = {1, -3, -2, -4};

Plane Surface(1) = {1};

Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} {Surface{1}; }
Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} {Surface{21};}
Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} {Surface{38};}
Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} {Surface{55};}


Physical Volume ("Myocardium") = {1, 2, 3, 4};
Physical Surface ("Basal plane") = {37, 54, 71, 20};
Physical Surface ("Endocardium") = {29, 12, 46, 63};
Physical Surface ("Epicardium") = {16, 33, 50, 67};

Mesh 3;