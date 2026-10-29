

** Beginning of compilation **


** End of compilation **


CATALOG VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  V	2	3
 3  B	4	5

OBSERVATIONAL VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  TB	3	INDEF
 2  XB	4	INDEF
 3  xB	5	INDEF
 4  yB	6	INDEF
 5  mB	7	8
 7  TV	10	INDEF
 8  XV	11	INDEF
 9  xV	12	INDEF
10  yV	13	INDEF
11  mV	14	15

FIT AND CONSTANT PARAMETER VALUES:

 1  v1	25.	
 2  v2	-0.03	
 3  b1	2.1	
 4  b2	-0.06	
 5  v3	0.18	
 6  b3	0.3	

TRANSFORMATION EQUATIONS:

 1 VFIT:  mV = V+v1+v2*(B-V)+v3*XV
   delta(VFIT, v1) = 0.1
   delta(VFIT, v2) = 0.1
   delta(VFIT, v3) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 

 2 BFIT:  mB = B+b1+b2*(B-V)+b3*XB
   delta(BFIT, b1) = 0.1
   delta(BFIT, b2) = 0.1
   delta(BFIT, b3) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 


Catalog input variables         = 4
First catalog column            = 2
Last catalog column             = 5

Observational input variables   = 12
First observational column      = 3
Last observational column       = 15

Fitting parameters              = 6
Constant parameters             = 0

Auxiliary (set) equations       = 0
Transformation equations        = 2

Warnings                        = 0
Errors                          = 0
