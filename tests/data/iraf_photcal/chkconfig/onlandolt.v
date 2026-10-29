

** Beginning of compilation **


** End of compilation **


CATALOG VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  V	4	12
 2  BV	5	13
 3  UB	6	14
 4  VR	7	15
 5  RI	8	16
 6  VI	9	17

OBSERVATIONAL VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  TB	3	INDEF
 2  XB	4	INDEF
 3  xB	5	INDEF
 4  yB	6	INDEF
 5  mB	7	8
 7  TI	10	INDEF
 8  XI	11	INDEF
 9  xI	12	INDEF
10  yI	13	INDEF
11  mI	14	15
13  TR	17	INDEF
14  XR	18	INDEF
15  xR	19	INDEF
16  yR	20	INDEF
17  mR	21	22
19  TU	24	INDEF
20  XU	25	INDEF
21  xU	26	INDEF
22  yU	27	INDEF
23  mU	28	29
25  TV	31	INDEF
26  XV	32	INDEF
27  xV	33	INDEF
28  yV	34	INDEF
29  mV	35	36

FIT AND CONSTANT PARAMETER VALUES:

 1  u1	0.	
 2  u2	0.65	
 3  u3	0.	
 4  u4	0.	(constant)
 5  b1	0.	
 6  b2	0.35	
 7  b3	0.	
 8  b4	0.	(constant)
 9  v1	0.	
10  v2	0.17	
11  v3	0.	
12  v4	0.	(constant)
13  r1	0.	
14  r2	0.08	
15  r3	0.	
16  r4	0.	(constant)
17  i1	0.	
18  i2	0.03	
19  i3	0.	
20  i4	0.	(constant)

TRANSFORMATION EQUATIONS:

 1 UFIT:  mU = (UB+BV+V)+u1+u2*XU+u3*UB+u4*UB*XU
   delta(UFIT, u1) = 0.1
   delta(UFIT, u2) = 0.1
   delta(UFIT, u3) = 0.1
   delta(UFIT, u4) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 

 2 BFIT:  mB = (BV+V)+b1+b2*XB+b3*BV+b4*BV*XB
   delta(BFIT, b1) = 0.1
   delta(BFIT, b2) = 0.1
   delta(BFIT, b3) = 0.1
   delta(BFIT, b4) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 

 3 VFIT:  mV = V+v1+v2*XV+v3*BV+v4*BV*XV
   delta(VFIT, v1) = 0.1
   delta(VFIT, v2) = 0.1
   delta(VFIT, v3) = 0.1
   delta(VFIT, v4) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 

 4 RFIT:  mR = (V-VR)+r1+r2*XR+r3*VR+r4*VR*XR
   delta(RFIT, r1) = 0.1
   delta(RFIT, r2) = 0.1
   delta(RFIT, r3) = 0.1
   delta(RFIT, r4) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 

 5 IFIT:  mI = (V-VI)+i1+i2*XI+i3*VI+i4*VI*XI
   delta(IFIT, i1) = 0.1
   delta(IFIT, i2) = 0.1
   delta(IFIT, i3) = 0.1
   delta(IFIT, i4) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 


Catalog input variables         = 12
First catalog column            = 4
Last catalog column             = 17

Observational input variables   = 30
First observational column      = 3
Last observational column       = 36

Fitting parameters              = 15
Constant parameters             = 5

Auxiliary (set) equations       = 0
Transformation equations        = 5

Warnings                        = 0
Errors                          = 0
