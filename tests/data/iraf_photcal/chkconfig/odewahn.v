

** Beginning of compilation **


** End of compilation **


CATALOG VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  V	2	3
 3  BV	4	5
 5  VR	6	7

OBSERVATIONAL VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  TB	3	INDEF
 2  XB	4	INDEF
 3  xB	5	INDEF
 4  yB	6	INDEF
 5  mB	7	8
 7  TR	10	INDEF
 8  XR	11	INDEF
 9  xR	12	INDEF
10  yR	13	INDEF
11  mR	14	15
13  TV	17	INDEF
14  XV	18	INDEF
15  xV	19	INDEF
16  yV	20	INDEF
17  mV	21	22

FIT AND CONSTANT PARAMETER VALUES:

 1  b1	0.	
 2  b2	0.35	
 3  b3	0.	
 4  b4	0.	
 5  v1	0.	
 6  v2	0.17	
 7  v3	0.	
 8  r1	0.	
 9  r2	0.08	
10  r3	0.	

TRANSFORMATION EQUATIONS:

 1 BFIT:  mB = (BV+V)+b1+b2*XB+b3*BV+b4*BV*XB
   delta(BFIT, b1) = 0.1
   delta(BFIT, b2) = 0.1
   delta(BFIT, b3) = 0.1
   delta(BFIT, b4) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 

 2 VFIT:  mV = V+v1+v2*XV+v3*BV
   delta(VFIT, v1) = 0.1
   delta(VFIT, v2) = 0.1
   delta(VFIT, v3) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 

 3 RFIT:  mR = (V-VR)+r1+r2*XR+r3*VR
   delta(RFIT, r1) = 0.1
   delta(RFIT, r2) = 0.1
   delta(RFIT, r3) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 


Catalog input variables         = 6
First catalog column            = 2
Last catalog column             = 7

Observational input variables   = 18
First observational column      = 3
Last observational column       = 22

Fitting parameters              = 10
Constant parameters             = 0

Auxiliary (set) equations       = 0
Transformation equations        = 3

Warnings                        = 0
Errors                          = 0
