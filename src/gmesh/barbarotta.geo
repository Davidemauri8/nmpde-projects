SetFactory("OpenCASCADE");

// Parameters
d = 29.1;
xi_endo = 0.6;
xi_epi = 1.02;
z_cut = 11.9; 

a_endo = d * Sinh(xi_endo);
c_endo = d * Cosh(xi_endo);
a_epi = d * Sinh(xi_epi);
c_epi = d * Cosh(xi_epi);


Sphere(1) = {0, 0, 0, 1}; 
Sphere(2) = {0, 0, 0, 1}; 

Dilate {{0, 0, 0}, {a_epi, a_epi, c_epi}} { Volume{1}; }
Dilate {{0, 0, 0}, {a_endo, a_endo, c_endo}} { Volume{2}; }


BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

Translate {0, 0, z_cut} { Volume{3}; }

Box(4) = {-c_epi, -c_epi, -c_epi, 2*c_epi, 2*c_epi, c_epi};
BooleanDifference{ Volume{3}; Delete; }{ Volume{4}; Delete; }


Physical Surface("Dirichlet") = {2}; 
Physical Surface("Neumann") = {3};   
Physical Surface("Robin") = {1}; 
Physical Volume("Heart") = {3};

Mesh.CharacteristicLengthMax = 1.5;
Mesh 3;