//+
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 0.6};
//+
Point(2) = {27, 0, 0, 0.6};
//+
Point(3) = {0, 27, 0, 0.6};
//+
Point(4) = {27, 27, 0, 0.6};
//+
Line(1) = {3, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 4};
//+
Line(4) = {4, 3};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Extrude {0, 0, 27} {
  Surface{1}; 
}
//+
Transfinite Curve {12, 7, 9, 11, 8, 6, 5, 10, 1, 2, 3, 4} = 28 Using Progression 1;
//+
Transfinite Surface {6};
//+
Transfinite Surface {2};
//+
Transfinite Surface {1};
//+
Transfinite Surface {4};
//+
Transfinite Surface {3};
//+
Transfinite Surface {5};
//+
Recombine Surface {6, 2, 1, 4, 3, 5};
//+
Transfinite Volume{1} = {6, 7, 8, 5, 1, 2, 4, 3};
//+
Physical Surface("boundary") = {5, 6, 3, 4, 1, 2};
//+
Physical Volume("boundary") += {1};
