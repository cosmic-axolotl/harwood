

** Beginning of compilation **


** End of compilation **


CATALOG VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  V	2	INDEF

OBSERVATIONAL VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  mV	3	INDEF
 2  XV	4	INDEF

FIT AND CONSTANT PARAMETER VALUES:

 1  v1	25.	
 2  v2	0.1	

TRANSFORMATION EQUATIONS:

 1 VFIT:  mV = V+v1+v2*XV
   delta(VFIT, v1) = 0.1
   delta(VFIT, v2) = 0.1
   error = 0.01,  min = ,  max = 0.1
   weight = 1,  min = 0.5,  max = 
   plot  x = ,  y = 


Catalog input variables         = 1
First catalog column            = 2
Last catalog column             = 2

Observational input variables   = 2
First observational column      = 3
Last observational column       = 4

Fitting parameters              = 2
Constant parameters             = 0

Auxiliary (set) equations       = 0
Transformation equations        = 1

Warnings                        = 0
Errors                          = 0
