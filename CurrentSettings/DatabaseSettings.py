DatabaseName = 'Default_database'

Boundary_conditions = ['Pm', 'Po', 'Ps', 'hm', 'hs']

import CurrentSettings.CFDSettings
ModelParameters = []
if CurrentSettings.CFDSettings.model == 'DHRM':
    ModelParameters = ['theta_0', 'a', 'b', 'dh', 'Psintersect']

Geometry_parameters= ['LintletConst', 'DmotiveIn' , 'DmotiveOut' , 'Dthroat' , 'alphaMotiveDiff', 'alphaMotiveConv', 'Lmch','alphaSuction', 'Lsuction', 'Dmix', 'Lmix', 'alphadiff', 'DdiffOut', 'Loutlet', 'Dsuc', 'ThicknessNozzle', 'ThicknessNozzleOuter']
Indicators = ['SimulatedIndicator','ConvergenceIndicator','CrashIndicator','Experiment_model_type','MeshRefinement']
Outputs = ['mfr_m','mfr_s','mfr_o','mfr_err', 'uni_alpha', 'uni_vel', 'ds1', 'eff']

Parameters = Boundary_conditions + ModelParameters + Geometry_parameters
All = Boundary_conditions + ModelParameters + Geometry_parameters + Indicators+  Outputs

Database_from_testpointMatrix = True
