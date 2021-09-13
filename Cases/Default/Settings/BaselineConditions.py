Pm = 8e+6
Po =  5e+6
Ps = 4e+6
hm =3e+5
hs =4.4e+5 

Boundary_conditions = [Pm, Po, Ps, hm, hs]

LintletConst =0.01
DmotiveIn = 0.05 
DmotiveOut = 0.00781
Dthroat = 0.00572
alphaMotiveDiff = 4.0
alphaMotiveConv = 30.0
Lmch = 0.0095
alphaSuction = 40
Lsuction = 0.05
Dmix = 0.0125
Lmix = 0.13
alphadiff = 5
DdiffOut = 0.05
Loutlet = 0.01
Dsuc= -1 #-1 indicates calculate based on internal params
ThicknessNozzle = 0.001
ThicknessNozzleOuter= 0.01

Geometry_parameters= [LintletConst, DmotiveIn , DmotiveOut , Dthroat , alphaMotiveDiff, alphaMotiveConv, Lmch,alphaSuction, Lsuction, Dmix, Lmix, alphadiff, DdiffOut, Loutlet, Dsuc, ThicknessNozzle, ThicknessNozzleOuter]

InputStandard = Boundary_conditions + Geometry_parameters 
