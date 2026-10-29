

** Beginning of compilation **


** End of compilation **


CATALOG VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  V	2	INDEF

OBSERVATIONAL VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  mV	3	INDEF

FIT AND CONSTANT PARAMETER VALUES:

 1  v1	25.	

TRANSFORMATION EQUATIONS:

 1 VFIT:  mV = V+v1
   delta(VFIT, v1) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 

 2 VFIT2:  mV = V+v1+v1*mV*V
   delta(VFIT2, v1) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 


Catalog input variables         = 1
First catalog column            = 2
Last catalog column             = 2

Observational input variables   = 1
First observational column      = 3
Last observational column       = 3

Fitting parameters              = 1
Constant parameters             = 0

Auxiliary (set) equations       = 0
Transformation equations        = 2

Warnings                        = 0
Errors                          = 0
