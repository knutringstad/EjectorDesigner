Pm = 9e+6
Po =  4.2e+6
Ps = 15e+6
hm =3.12e+5
hs =1.32e+5 

Boundary_conditions = [Pm, Po, Ps, hm, hs]

factor = 1.3

LintletConst =0.01
DmotiveIn = 0.01 
DmotiveOut = 0.004
Dthroat = 0.00283
alphaMotiveDiff = 6.0
alphaMotiveConv = 30.0
Lmch = 0.004
alphaSuction = 20
Lsuction = 0.02
Dmix = 0.00489
Lmix = 0.016
alphadiff = 20
DdiffOut = 0.025
Loutlet = 0.01
Dsuc= -1 #-1 indicates calculate based on internal params
ThicknessNozzle = 0.00035
ThicknessNozzleOuter= 0.01

Geometry_parameters= [LintletConst, DmotiveIn , DmotiveOut , Dthroat , alphaMotiveDiff, alphaMotiveConv, Lmch,alphaSuction, Lsuction, Dmix, Lmix, alphadiff, DdiffOut, Loutlet, Dsuc, ThicknessNozzle, ThicknessNozzleOuter]

InputStandard = Boundary_conditions + Geometry_parameters 
