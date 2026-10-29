

** Beginning of compilation **

** Warning near line 1: catalog ;
The catalog section is empty

** End of compilation **


OBSERVATIONAL VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  mV	3	INDEF

FIT AND CONSTANT PARAMETER VALUES:

 1  v1	25.	
 2  v2	150.	
 3  v3	0.5	
 4  v4	-2.	
 5  v5	3.	
 6  c1	1.	(constant)

TRANSFORMATION EQUATIONS:

 1 VFIT:  mV = sqrt(abs(v1))+log10(v2)*exp(-v3)+sin(v4)**2**2--v5/+c1+atan(mV)+log(cos(tan(asin(acos(mV)))))
   delta(VFIT, v1) = 0.1
   delta(VFIT, v2) = 0.1
   delta(VFIT, v3) = 0.1
   delta(VFIT, v4) = 0.1
   delta(VFIT, v5) = 0.1
   delta(VFIT, c1) = 0.1
   error = ,  min = ,  max = 
   weight = ,  min = ,  max = 
   plot  x = ,  y = 


Catalog input variables         = 0
First catalog column            = INDEF
Last catalog column             = INDEF

Observational input variables   = 1
First observational column      = 3
Last observational column       = 3

Fitting parameters              = 5
Constant parameters             = 1

Auxiliary (set) equations       = 0
Transformation equations        = 1

Warnings                        = 1
Errors                          = 0
