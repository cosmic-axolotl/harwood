"""The cursor files of imexamine's other keys, in tests/iraf_field.field()'s stars."""

STAR, OTHER = "41.0 45.0", "110.0 53.0"

CASES = {
    "positions": f"""{STAR} 1 x
20.5 30.25 1 x
70.0 60.0 1 y
{STAR} 1 x
20.5 80.25 1 x
100.0 20.0 1 x
0.0 0.0 1 y
33.0 44.0 1 x
12.4 18.6 1 b
40.5 7.5 1 b
40.0 7.0 1 b
12.0 18.0 1 b
""",
    "quick": f"""{STAR} 1 ,
{OTHER} 1 ,
20.0 20.0 1 ,
161.0 141.0 1 ,
{STAR} 1 m
{OTHER} 1 ,
""",
    "quick-g": f"""{STAR} 1 :fittype gaussian
{STAR} 1 ,
{OTHER} 1 ,
161.0 141.0 1 ,
{STAR} 1 :radius 3
{STAR} 1 ,
{STAR} 1 :rplot 2
{STAR} 1 ,
""",
    "quick-m": f"""{STAR} 1 :beta 2.5
{STAR} 1 ,
{OTHER} 1 :radius 7
{OTHER} 1 ,
{OTHER} 1 :background no
{OTHER} 1 ,
{OTHER} 1 :center no
{OTHER} 1 ,
{STAR} 1 :magzero 20
{STAR} 1 ,
""",
    "section": f"""{STAR} 1 t
{OTHER} 1 :ncoutput 11
{OTHER} 1 :nloutput 8
{OTHER} 1 t
20.3 20.7 1 t
""",
    "log": f"""{STAR} 1 x
1.0 1.0 1 w
{OTHER} 1 x
1.0 1.0 1 w
20.0 20.0 1 x
""",
    "vectors": f"""10.0 10.0 1 v
80.0 60.0 1 v
10.0 20.0 1 v
80.0 20.0 1 v
30.0 10.0 1 v
30.0 60.0 1 v
80.0 60.0 1 v
10.0 10.0 1 v
30.5 15.25 1 v
62.75 40.5 1 v
40.0 40.0 1 u
60.0 55.0 1 u
40.0 40.0 1 u
60.0 40.0 1 u
40.0 40.0 1 u
40.0 55.0 1 u
{STAR} 1 :naverage 3
10.0 10.0 1 v
80.0 60.0 1 v
10.0 20.0 1 v
80.0 20.0 1 v
30.0 10.0 1 v
30.0 60.0 1 v
{STAR} 1 :naverage 4
40.0 40.0 1 u
60.0 55.0 1 u
{STAR} 1 :naverage 1
190.0 20.0 1 v
260.0 80.0 1 v
""",
    "over": f"""{STAR} 1 l
1.0 1.0 1 o
{OTHER} 1 l
{STAR} 1 c
1.0 1.0 1 o
10.0 10.0 1 v
80.0 60.0 1 v
1.0 1.0 1 f
{STAR} 1 .
{OTHER} 1 .
1.0 1.0 1 f
""",
    "boundary": f"""{STAR} 1 :naverage 5
{STAR} 1 v
10.0 3.0 1 v
10.0 30.0 1 v
2.0 10.0 1 v
40.0 10.0 1 v
3.0 3.0 1 v
30.0 20.0 1 v
""",
}
