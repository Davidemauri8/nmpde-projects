// --------------------------------------------------------
// Cardiac Ventricle Geometry (Dimensions in meters)
// --------------------------------------------------------

mesh_res = 0.025; // h

// Geometry thresholds for the truncated ellipsoid
phi_endo = Acos(5 / 17.0);  
phi_epi  = Acos(5 / 20.0);  

// Radial and Longitudinal semi-axes
rad_inner = 0.007; 
lon_inner = 0.017; 
rad_outer = 0.010; 
lon_outer = 0.020;  

// --------------------------------------------------------
// Macro to generate points on the ellipsoidal shell
// --------------------------------------------------------

Function GenerateVentriclePoint
    Point(p_id) = { axis_s * Sin(p_mu) * Cos(p_theta),
                    axis_s * Sin(p_mu) * Sin(p_theta),
                    -axis_l * Cos(p_mu),
                    mesh_res };
Return

// --------------------------------------------------------
// Build Reference Points
// --------------------------------------------------------

// Focal Center
p_origin = newp;
Point(p_origin) = {0.0, 0.0, 0.0};

// Inner Wall Construction
p_theta = 0.0;
axis_s = rad_inner;
axis_l = lon_inner;

p_mu = 0.0;
p_apex_in = newp;
p_id = p_apex_in;
Call GenerateVentriclePoint;

p_mu = phi_endo;
p_base_in = newp;
p_id = p_base_in;
Call GenerateVentriclePoint;

// Outer Wall Construction
axis_s = rad_outer;
axis_l = lon_outer;

p_mu = 0.0;
p_apex_out = newp;
p_id = p_apex_out;
Call GenerateVentriclePoint;

p_mu = phi_epi;
p_base_out = newp;
p_id = p_base_out;
Call GenerateVentriclePoint;

// --------------------------------------------------------
// Define Cross-Sectional Lines
// --------------------------------------------------------

Ellipse(1) = {p_base_in, p_origin, p_apex_in, p_apex_in};
Ellipse(2) = {p_base_out, p_origin, p_apex_out, p_apex_out};
Line(3)    = {p_apex_out, p_apex_in};
Line(4)    = {p_base_in, p_base_out};

// Create the 2D shell profile
Curve Loop(1) = {1, -3, -2, -4};
Plane Surface(1) = {1};

// --------------------------------------------------------
// Rotational Extrusion to form the 3D Volume
// --------------------------------------------------------

// Extrude in 90-degree increments
Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} { Surface{1}; }
Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} { Surface{21}; }
Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} { Surface{38}; }
Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} { Surface{55}; }

// --------------------------------------------------------
// Physical Groups & Final Translation
// --------------------------------------------------------

Physical Volume("Myocardium")   = {1, 2, 3, 4};
Physical Surface("Basal plane") = {37, 54, 71, 20};
Physical Surface("Endocardium") = {29, 12, 46, 63};
Physical Surface("Epicardium")  = {16, 33, 50, 67};

// Shift the entire geometry 0.01 units upwards along Z
Translate {0, 0, 0.01} { Volume{1, 2, 3, 4}; }

Mesh 3;