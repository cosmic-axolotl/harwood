

** Beginning of compilation **


** End of compilation **


CATALOG VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  V	2	3
 3  B	4	5

OBSERVATIONAL VARIABLES, COLUMNS, AND ERROR COLUMNS:

 1  TB	3	INDEF
 2  XB	4	INDEF
 3  mB	7	8
 5  TV	10	INDEF
 6  XV	11	INDEF
 7  mV	14	15

FIT AND CONSTANT PARAMETER VALUES:

 1  v1	25.	
 2  v2	-0.03	
 3  v3	0.18	
 4  v4	0.01	(constant)
 5  b1	2.1	
 6  b2	-0.06	
 7  b3	0.3	

AUXILIARY (SET) EQUATIONS:

 1 BV = B-V
   error = ,  min = ,  max = 


TRANSFORMATION EQUATIONS:

 1 VFIT:  mV = V+v1+v2*BV+v3*XV+v4*XV*BV
   delta(VFIT, v1) = 0.1
   delta(VFIT, v2) = 0.05
   derivative (VFIT, v3) = XV
   delta(VFIT, v4) = 0.1
   error = 0.01+0.005*(XV-1.0),  min = ,  max = 
   weight = 1.0/(0.0004+0.001*XV),  min = ,  max = 
   plot  x = ,  y = 

 2 BFIT:  mB = B+b1+b2*BV+b3*XB
   delta(BFIT, b1) = 0.1
   delta(BFIT, b2) = 0.1
   delta(BFIT, b3) = 0.1
   error = 0.012,  min = ,  max = 
   weight = 1.0/(0.0003+abs(mB+7.0)*0.0001),  min = 1500.0,  max = 2500.0
   plot  x = ,  y = 


Catalog input variables         = 4
First catalog column            = 2
Last catalog column             = 5

Observational input variables   = 8
First observational column      = 3
Last observational column       = 15

Fitting parameters              = 6
Constant parameters             = 1

Auxiliary (set) equations       = 1
Transformation equations        = 2

Warnings                        = 0
Errors                          = 0
