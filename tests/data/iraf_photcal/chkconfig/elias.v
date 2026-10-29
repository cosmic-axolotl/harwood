

** Beginning of compilation **


** End of compilation **


CATALOG VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  K	2	3
 3  JK	4	5
 5  HK	6	7
 7  KL	8	9

OBSERVATIONAL VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  TH	3	INDEF
 2  XH	4	INDEF
 3  xH	5	INDEF
 4  yH	6	INDEF
 5  mH	7	8
 7  TJ	10	INDEF
 8  XJ	11	INDEF
 9  xJ	12	INDEF
10  yJ	13	INDEF
11  mJ	14	15
13  TK	17	INDEF
14  XK	18	INDEF
15  xK	19	INDEF
16  yK	20	INDEF
17  mK	21	22
19  TL	24	INDEF
20  XL	25	INDEF
21  xL	26	INDEF
22  yL	27	INDEF
23  mL	28	29

FIT AND CONSTANT PARAMETER VALUES:

 1  j1	0.	
 2  j2	0.05	
 3  j3	0.	
 4  h1	0.	
 5  h2	0.05	
 6  h3	0.	
 7  k1	0.	
 8  k2	0.05	
 9  k3	0.	
10  l1	0.	
11  l2	0.05	
12  l3	0.	

TRANSFORMATION EQUATIONS:

 1 JFIT:  mJ = (K+JK)+j1+j2*XJ+j3*JK
   delta(JFIT, j1) = 0.1
   delta(JFIT, j2) = 0.1
   delta(JFIT, j3) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 

 2 HFIT:  mH = (K+HK)+h1+h2*XH+h3*HK
   delta(HFIT, h1) = 0.1
   delta(HFIT, h2) = 0.1
   delta(HFIT, h3) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 

 3 KFIT:  mK = K+k1+k2*XK+k3*HK
   delta(KFIT, k1) = 0.1
   delta(KFIT, k2) = 0.1
   delta(KFIT, k3) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 

 4 LFIT:  mL = (K-KL)+l1+l2*XL+l3*KL
   delta(LFIT, l1) = 0.1
   delta(LFIT, l2) = 0.1
   delta(LFIT, l3) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 


Catalog input variables         = 8
First catalog column            = 2
Last catalog column             = 9

Observational input variables   = 24
First observational column      = 3
Last observational column       = 29

Fitting parameters              = 12
Constant parameters             = 0

Auxiliary (set) equations       = 0
Transformation equations        = 4

Warnings                        = 0
Errors                          = 0
