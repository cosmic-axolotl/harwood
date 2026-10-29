

** Beginning of compilation **

** Warning near line 60: Duplicate column for input variables [Ks] and [K]

** End of compilation **


CATALOG VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  J	2	3
 3  H	4	5
 5  K	6	7
 7  Ks	6	7

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
19  TKs	24	INDEF
20  XKs	25	INDEF
21  xKs	26	INDEF
22  yKs	27	INDEF
23  mKs	28	29

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
10  ks1	0.	
11  ks2	0.05	
12  ks3	0.	

TRANSFORMATION EQUATIONS:

 1 JFIT:  mJ = J+j1+j2*XJ+j3*(J-K)
   delta(JFIT, j1) = 0.1
   delta(JFIT, j2) = 0.1
   delta(JFIT, j3) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 

 2 HFIT:  mH = H+h1+h2*XH+h3*(H-K)
   delta(HFIT, h1) = 0.1
   delta(HFIT, h2) = 0.1
   delta(HFIT, h3) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 

 3 KFIT:  mK = K+k1+k2*XK+k3*(H-K)
   delta(KFIT, k1) = 0.1
   delta(KFIT, k2) = 0.1
   delta(KFIT, k3) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 

 4 KSFIT:  mKs = Ks+ks1+ks2*XKs+ks3*(H-Ks)
   delta(KSFIT, ks1) = 0.1
   delta(KSFIT, ks2) = 0.1
   delta(KSFIT, ks3) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 


Catalog input variables         = 8
First catalog column            = 2
Last catalog column             = 7

Observational input variables   = 24
First observational column      = 3
Last observational column       = 29

Fitting parameters              = 12
Constant parameters             = 0

Auxiliary (set) equations       = 0
Transformation equations        = 4

Warnings                        = 1
Errors                          = 0
