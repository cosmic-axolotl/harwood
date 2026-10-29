

** Beginning of compilation **

** Warning near line 16: Parameter delta and derivative defined for [b1] in equation [BFIT]
** Warning near line 16: Parameter delta and derivative defined for [v2] in equation [BFIT]
** Warning near line 16: Parameter delta and derivative defined for [v2] in equation [VFIT]

** End of compilation **


CATALOG VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  V	2	INDEF
 2  B	3	INDEF

OBSERVATIONAL VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  mV	3	INDEF
 2  mB	5	INDEF
 3  XV	4	INDEF

FIT AND CONSTANT PARAMETER VALUES:

 1  v1	25.	
 2  v2	1.	
 3  b1	1.	

TRANSFORMATION EQUATIONS:

 1 VFIT:  mV = V+v1+v2*XV
   delta(VFIT, v1) = 0.1
   derivative (VFIT, v2) = XV
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 

 2 BFIT:  mB = B+b1+v2*XV
   derivative (BFIT, b1) = 1
   derivative (BFIT, v2) = XV
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 


Catalog input variables         = 2
First catalog column            = 2
Last catalog column             = 3

Observational input variables   = 3
First observational column      = 3
Last observational column       = 5

Fitting parameters              = 3
Constant parameters             = 0

Auxiliary (set) equations       = 0
Transformation equations        = 2

Warnings                        = 3
Errors                          = 0
