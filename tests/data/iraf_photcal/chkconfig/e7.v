

** Beginning of compilation **


** End of compilation **


CATALOG VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  V	2	INDEF
 2  B	3	INDEF

OBSERVATIONAL VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  mV	3	INDEF
 2  XV	4	INDEF

FIT AND CONSTANT PARAMETER VALUES:

 1  v1	25.	
 2  v2	0.1	
 3  v3	0.2	(constant)

AUXILIARY (SET) EQUATIONS:

 1 BV = B-V
   error = XV+V,  min = ,  max = 


TRANSFORMATION EQUATIONS:

 1 VFIT:  mV = V+v1+v2*XV+v3*BV
   delta(VFIT, v1) = 0.1
   delta(VFIT, v2) = 0.1
   delta(VFIT, v3) = 0.1
   error = v1*2,  min = 0.0,  max = XV
   weight = mV+V,  min = v1,  max = 2
   plot  x = mV,  y = V


Catalog input variables         = 2
First catalog column            = 2
Last catalog column             = 3

Observational input variables   = 2
First observational column      = 3
Last observational column       = 4

Fitting parameters              = 2
Constant parameters             = 1

Auxiliary (set) equations       = 1
Transformation equations        = 1

Warnings                        = 0
Errors                          = 0
