SetFactory("OpenCASCADE");

// -------------------------------
// Parametri geometrici (realistici LV)
// -------------------------------
epicardiumWallRadius = 27;         // semiasse epicardico trasverso (mm)
endocardiumWallRadius = 18;        // epicardio - spessore parete (~9 mm)

a = 1;                             // scala x,y (tono ellissoide)
c = 1.67;                          // scala z (rapporto longitudinale LV)

heightCut = 20;                    // taglio basale (mm)
innerHoleRadius = 6;               // foro assiale (tract)

meshFineness = 3;                  // finezza mesh

// -------------------------------
// Guscio ellissoidale (LV)
// -------------------------------
Sphere(1) = {0, 0, 0, epicardiumWallRadius, 0};
Sphere(2) = {0, 0, 0, endocardiumWallRadius, 0};

// Allungo in z per ottenere un ellissoide realistico
Dilate{{0,0,0},{a,a,c}} { Volume{1}; }
Dilate{{0,0,0},{a,a,c}} { Volume{2}; }

// Guscio = epicardio - endocardio
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }  // --> Volume 3

// -------------------------------
// Taglio inferiore (tronco LV)
// -------------------------------
// Piano z = -heightCut
Plane(10) = {0, 0, -heightCut, 0};

BooleanIntersection{ Volume{3}; Delete; }{ Volume{10}; Delete; }  // --> Volume 11

// -------------------------------
// Foro cilindrico assiale
// -------------------------------
cylLength = 2 * heightCut;

Cylinder(20) = {0,0,-heightCut, 0,0,cylLength, innerHoleRadius, 0};

BooleanDifference{ Volume{11}; Delete; }{ Volume{20}; Delete; }  // --> Volume 30

// -------------------------------
// Physical groups
// -------------------------------
Physical Volume("Heart") = {30};

Physical Surface("Robin") = {8};
Physical Surface("Dirichlet") = {9};
Physical Surface("Neumann") = {7};

Physical Volume("Heart") = {1}; 

// -------------------------------
// Mesh
// -------------------------------
Mesh.CharacteristicLengthMax = meshFineness;
Mesh.CharacteristicLengthMin = meshFineness/2;

Mesh 3;
