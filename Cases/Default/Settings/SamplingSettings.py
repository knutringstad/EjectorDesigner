samplingBool = False
NumberOfSamples = 10
samplingMode = 'LinearEvenMinMaxSampling'  #'LinearLatinHyperCube' , 'LinearEvenMinMaxSampling' 

featureDict = {
    "Pm":0,
    "Ps":0,
    "Po":0,
    "hm":0,
    "hs":0,
    "LintletConst":0,
    "DmotiveIn":0,
    "DmotiveOut":0,
    "Dthroat":0,
    "alphaMotiveDiff":0,
    "alphaMotiveConv":0,
    "Lmch":0,
    "alphaSuction":0,
    "Lsuction":0,
    "Dmix":0,
    "Lmix":0,
    "alphadiff":0,
    "DdiffOut":0,
    "Loutlet":0,
    "Dsuc":0,
    "ThicknessNozzle":0,
    "ThicknessNozzleOuter":0,
}

FeatureNameList = [key for key, value in featureDict.items() if value == 1 ]

SamplingMinDict = {
    "Pm":0.5e+6,
    "Ps":0.5e+6,
    "Po":0.5e+6,
    "hm":2e+5,
    "hs":2e+5,
    "LintletConst":1e-3,
    "DmotiveIn":1e-3,
    "DmotiveOut":1e-3,
    "Dthroat":1e-3,
    "alphaMotiveDiff":1e-3,
    "alphaMotiveConv":1e-3,
    "Lmch":1e-3,
    "alphaSuction":1e-3,
    "Lsuction":1e-3,
    "Dmix":1e-3,
    "Lmix":1e-3,
    "alphadiff":0.01,
    "DdiffOut":1e-3,
    "Loutlet":1e-3,
    "Dsuc":1e-3,
    "ThicknessNozzle":1e-3,
    "ThicknessNozzleOuter":1e-3,
}


SamplingMaxDict = {
    "Pm":1.2e+7,
    "Ps":1.2e+7,
    "Po":1.2e+7,
    "hm":4.6e+5,
    "hs":4.6e+5,
    "LintletConst":20e-3,
    "Lmch":20e-3,
    "DmotiveIn":20e-3,
    "DmotiveOut":20e-3,
    "Dthroat":20e-3,
    "alphaMotiveDiff":90,
    "alphaMotiveConv":90,
    "Lmch":20e-3,
    "alphaSuction":90,
    "Lsuction":20e-3,
    "Dmix":20e-3,
    "Lmix":20e-3,
    "alphadiff":90,
    "DdiffOut":20e-3,
    "Loutlet":20e-3,
    "Dsuc":20e-3,
    "ThicknessNozzle":20e-3,
    "ThicknessNozzleOuter":20e-3,
}


if samplingMode == 'LinearEvenMinMaxSampling':
    #For LinearEvenMinMaxSamping specify N_f samples for each feature sampled
    #Sum N_f must equal Nsamples

    NLinearEvenSamplesDict = {
        "Pm":10,
        "Ps":0,
        "Po":0,
        "hm":0,
        "hs":0,
        "LintletConst":0,
        "DmotiveIn":0,
        "DmotiveOut":0,
        "Dthroat":0,
        "alphaMotiveDiff":0,
        "alphaMotiveConv":0,
        "Lmch":0,
        "alphaSuction":0,
        "Lsuction":0,
        "Dmix":0,
        "Lmix":0,
        "alphadiff":0,
        "DdiffOut":0,
        "Loutlet":0,
        "Dsuc":0,
        "ThicknessNozzle":0,
        "ThicknessNozzleOuter":0,
    }


    NLinearEvenSamplesList = list((NLinearEvenSamplesDict.values()))
    for i in range(len(NLinearEvenSamplesList)): NLinearEvenSamplesList[i]=int(NLinearEvenSamplesList[i])
    if not sum(NLinearEvenSamplesList) == NumberOfSamples:
        raise Exception("Number of samples must total to number of samples, Nsamples : %d, NsumSamplingList : %d" %( NumberOfSamples,sum(NLinearEvenSamplesList)))

import CurrentSettings.DatabaseSettings
if len(featureDict) < len(CurrentSettings.DatabaseSettings.Boundary_conditions)+len(CurrentSettings.DatabaseSettings.Geometry_parameters):
    print('ERROR IN SETUP: features vector must match boundary conditions and geometry parameter length')