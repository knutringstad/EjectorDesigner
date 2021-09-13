from pyDOE import *
def sample(N,featureDict,stdInput,params):
    import CurrentSettings.SamplingSettings

    if CurrentSettings.SamplingSettings.samplingMode == 'LinearLatinHyperCube':
        samples= sampling_linear_LHC(N,featureDict,stdInput,params)
    elif CurrentSettings.SamplingSettings.samplingMode == 'LinearEvenMinMaxSampling':
        samples= sampling_linear_even(N,featureDict,stdInput,params)
    elif CurrentSettings.SamplingSettings.samplingMode == 'Baseline':
        samples= sampling_baseline(N,featureDict,stdInput,params)
    else:
        raise Exception("No such sampling mode available. Check SamplingSettings.")

    return samples


def sampling_linear_LHC(N,featureDict,stdInput,params):
    # N is number of samples
    # features is vector of features to sample
    # std input is the input vector that takes the location of all non-features
    
    import csv
    import numpy as np
    import pandas as pd
    import CurrentSettings.DatabaseSettings
    import CurrentSettings.BaselineConditions
    import CurrentSettings.SamplingSettings
    import math
    
    Nfeatures = CurrentSettings.SamplingSettings.Nfeatures
    FeatureNameList = CurrentSettings.SamplingSettings.FeatureNameList

    x = np.zeros(( N, len(stdInput)))
    for j in range(N):
            x[j] = stdInput

    if Nfeatures <1: #No sampling, return standard input
        return x

    d =lhs(Nfeatures,samples=N,criterion="cm")
    D = pd.DataFrame(data = d, columns=FeatureNameList)

    params = CurrentSettings.DatabaseSettings.Parameters
    X = pd.DataFrame(data = x, columns=params)






    # dependencies dmix first
    feature = "Dmix"
    X[feature] = np.multiply(d[:,D.columns.get_loc(feature)],(CurrentSettings.SamplingSettings.SamplingMaxDict[feature] - CurrentSettings.SamplingSettings.SamplingMinDict[feature])) + CurrentSettings.SamplingSettings.SamplingMinDict[feature]

    feature = "alphadiff"
    X[feature] = np.multiply(d[:,D.columns.get_loc(feature)],(CurrentSettings.SamplingSettings.SamplingMaxDict[feature] - CurrentSettings.SamplingSettings.SamplingMinDict[feature])) + CurrentSettings.SamplingSettings.SamplingMinDict[feature]

    feature = "Lmch"
    X[feature] = np.multiply(d[:,D.columns.get_loc(feature)],(CurrentSettings.SamplingSettings.SamplingMaxDict[feature] - CurrentSettings.SamplingSettings.SamplingMinDict[feature])) + CurrentSettings.SamplingSettings.SamplingMinDict[feature]

    alpha_s = CurrentSettings.BaselineConditions.alphaSuction
    Lmch = CurrentSettings.BaselineConditions.Lmch
    t = CurrentSettings.BaselineConditions.ThicknessNozzle
    dthroat = CurrentSettings.BaselineConditions.Dthroat

    for j in range(N):
        # feature = "DmotiveOut"
        # Dmch = 2*(   math.tan(math.radians(alpha_s/2))*Lmch + X["Dmix"][j] /2   )
        # dmoMax = Dmch*0.98 - 2*t
        # dmoMax = min(dmoMax, 1.5*dthroat)
        # X[feature][j] = np.multiply(d[j,D.columns.get_loc(feature)],(dmoMax - dthroat*1.01)) + dthroat*1.01
        feature = "Lmix"
        X[feature][j] = np.multiply(d[j,D.columns.get_loc(feature)],(10*X["Dmix"][j] - 0.01*X["Dmix"][j])) + 0.01*X["Dmix"][j]
        feature = "DdiffOut"
        X[feature][j] = np.multiply(d[j,D.columns.get_loc(feature)],((10*X["Dmix"][j] - 1.01*X["Dmix"][j]))) + 1.01*X["Dmix"][j]




    #IF NO DEPENDENCIES JUST RUN:
    # for i,feature in enumerate(FeatureNameList):
    #     X[feature] = np.multiply(d[:,i],(CurrentSettings.SamplingSettings.SamplingMaxDict[feature] - CurrentSettings.SamplingSettings.SamplingMinDict[feature]))

    return X.to_numpy()
    




def sampling_linear_even(N,featureDict,stdInput,params):
    # N is number of samples
    # features is vector of features to sample
    # std input is the input vector that takes the location of all non-features

    import CurrentSettings.SamplingSettings    
    import csv
    import numpy as np

    NsamplesDict = CurrentSettings.SamplingSettings.NLinearEvenSamplesDict
    
    featuresList = list((featureDict.values()))
    for i in range(len(featuresList)): featuresList[i]=int(featuresList[i])
    Ndim = sum(featuresList)


    #Make the Samples matrix and fill with standard input
    Samples = np.zeros((N,len(stdInput)))
    for i in range(N):
        Samples[i][:] = stdInput

    if Ndim<1:
        return Samples

    #Go through parameters, if its a features add the correct number of evenly sampled variables from minimum to maximum 
    for j,param in enumerate(params):
        if featureDict[param] == 1 :
            Nfeat = NsamplesDict[param]
            minfeat= CurrentSettings.SamplingSettings.SamplingMinDict[param]
            maxfeat= CurrentSettings.SamplingSettings.SamplingMaxDict[param]
            Samples[i:i+Nfeat][j] = np.linspace(minfeat,maxfeat,Nfeat)


    return Samples




def sampling_baseline(N,featureDict,stdInput,params):
    # N is number of samples
    # features is vector of features to sample
    # std input is the input vector that takes the location of all non-features

    import CurrentSettings.SamplingSettings    
    import csv
    import numpy as np


    #Make the Samples matrix and fill with standard input
    Samples = np.zeros((N,len(stdInput)))
    for i in range(N):
        Samples[i][:] = stdInput

    return Samples






def customDistribution_DthroatBased(Input_baseline, x,features):
    import numpy as np
    from scipy.stats.distributions import norm
    import math
    import CurrentSettings.DatabaseSettings

    #Assumes that Dthroat is given and not a featureDict.
    #From it we can find Dmix, and then the other geometry parameters
                        # Satt: Dthroat  = konstant                (Dm-in, t, alpha_s )

                        # Dthroat                           <   Dmix                <     10 Dthroat
                        # Dmix + liten delta *      <   Doutlet            <     10 Dmix
                        # 1 celle                              <  Lmix                 <     50 Dmix
                        # Dthroat + liten delta * <   Dmo                 <     f ( Pm … trippelpunkt ?)     evt x*Dthroat
                        # Geometri ( alpha_s, Dmo, Dmix, t  )  <  Lmch  < 20* Dthroat

                        # Liten delta *< Alpha_diff < 90
                        # Liten delta *<  Alpha_m-d < 60 
                        # Liten delta *<  Alpha_m-c < 90   

                        

                        # * Legger til liten delta: Hvis for eksempel   D_outlet = Dmix    og    alpha_diff >0   får vi umulig geometri    (Loutlet --> inf)

                    #Performance map, thermodynamics:     28 < Ps < 55 bar    ,   1< Plift <15 bar , 
            #BONUSES:
            #  alpha_conv < Alpha_suction 

    featureDict = CurrentSettings.DatabaseSettings.featureDict

    #Pm,Po,Ps,hm,hs,Lmotive,DmotiveIn,DmotiveOut,Dthroat,alphaMotiveDiff,alphaMotiveConv,ThicknessNozzle,Lmch,alphaSuction,Dmix,Lmix,alphadiff, DdiffOut,Loutlet
    if featureDict["Dthroat"] ==1:
        print("Dthroat cannot be a featureDict for this method")
        return

    y = Input_baseline.copy()

    Dthroat = Input_baseline[8]

    j = 14
    if featureDict["Dmix"]==1: #Dmix
        minFeat = Dthroat
        maxFeat = 10*Dthroat 
        Dmix = (maxFeat-minFeat)*x[j] + minFeat
    else:
        Dmix = Input_baseline[j]

    y[j] =Dmix
    
    j=15
    if featureDict["Lmix"]==1: #Lmix
        minFeat = 0.0001
        maxFeat = 50*Dmix 
        Lmix = (maxFeat-minFeat)*x[j] + minFeat
    else:
        Lmix = Input_baseline[j]
    y[j] =Lmix

    j=0
    if featureDict["Pm"]==1: #Pm
        minFeat = 7.5e+6
        maxFeat = 14e+6
        Pm = (maxFeat-minFeat)*x[j] + minFeat
    else:
        Pm = Input_baseline[j]
    y[j] =Pm

    j=2
    if featureDict["Ps"]==1: #Ps
        minFeat = 2.8e+6
        maxFeat = 5.5e+6
        Ps = (maxFeat-minFeat)*x[j] + minFeat
    else:
        Ps = Input_baseline[j]
    y[j] =Ps

    j=1
    if featureDict["Po"]==1: #Po
        minFeat = Ps+0.1e+6
        maxFeat = Ps+1.5e+6
        Po = (maxFeat-minFeat)*x[j] + minFeat
    else:
        Po = Input_baseline[j]
    y[j] =Po
    
    j=3
    if featureDict["hm"]==1: #hm
        minFeat = 250e+3
        maxFeat = 340e+3
        hm = (maxFeat-minFeat)*x[j] + minFeat
    else:
        hm = Input_baseline[j]
    y[j] =hm

    j=4
    if featureDict["hs"]==1: #hs
        minFeat = 380e+3
        maxFeat = 460e+3
        hs = (maxFeat-minFeat)*x[j] + minFeat
    else:
        hs = Input_baseline[j]
    y[j] =hs

    j=6
    if featureDict["DmotiveIn"]==1: #DmotiveIn
        minFeat = 0.03
        maxFeat = 0.06
        DmotiveIn = (maxFeat-minFeat)*x[j] + minFeat
    else:
        DmotiveIn = Input_baseline[j]
    y[j] =DmotiveIn

    j=9
    if featureDict["alphaMotiveDiff"]==1: #alphaMotiveDiff
        minFeat = 1
        maxFeat = 60
        alphaMotiveDiff = (maxFeat-minFeat)*x[j] + minFeat
    else:
        alphaMotiveDiff = Input_baseline[j]
    y[j] =alphaMotiveDiff

    j=11
    if featureDict["ThicknessNozzle"]==1: #thickness
        minFeat = 0.0001
        maxFeat = 0.001
        thickness = (maxFeat-minFeat)*x[j] + minFeat
    else:
        thickness = Input_baseline[j]
    y[j] =thickness

    j=13
    if featureDict["alphaSuction"]==1: #alphaSuction
        minFeat = 1
        maxFeat = 90
        alphaSuction = (maxFeat-minFeat)*x[j] + minFeat
    else:
        alphaSuction = Input_baseline[j]
    y[j] =alphaSuction

    j=10
    if featureDict["alphaMotiveConv"]==1: #alphaMotiveConv
        minFeat = 10
        maxFeat = alphaSuction
        alphaMotiveConv = (maxFeat-minFeat)*x[j] + minFeat
    else:
        alphaMotiveConv = Input_baseline[j]
    y[j] =alphaMotiveConv


    j=16
    if featureDict["alphadiff"]==1: #alphadiff
        minFeat = 1
        maxFeat = 90
        alphadiff = (maxFeat-minFeat)*x[j] + minFeat
    else:
        alphadiff = Input_baseline[j]
    y[j] =alphadiff

    j=17
    if featureDict["DdiffOut"]==1: #DdiffOut
        minFeat = Dmix+0.0001
        maxFeat = 10*Dmix
        DdiffOut = (maxFeat-minFeat)*x[j] + minFeat
    else:
        DdiffOut = Input_baseline[j]
    y[j] =DdiffOut

    j=18
    if featureDict["Loutlet"]==1: #Loutlet
        minFeat = DdiffOut
        maxFeat = 4*DdiffOut
        Loutlet = (maxFeat-minFeat)*x[j] + minFeat
    else:
        Loutlet = Input_baseline[j]
    y[j] =Loutlet

    j=7
    if featureDict["DmotiveOut"]==1: #DmotiveOut
        minFeat = Dthroat + 0.0001
        maxFeat = 3*Dthroat# f(Pm, trippelpoint...) #needs implementation
        DmotiveOut = (maxFeat-minFeat)*x[j] + minFeat
    else:
        DmotiveOut = Input_baseline[j]
    y[j] =DmotiveOut

    j=12
    minFeat = max (  (DmotiveOut/2 + thickness - Dmix/2)/( math.tan(alphaSuction/2)), 0.0001)
    maxFeat = 10*Dthroat

    if featureDict["Lmch"]==1: #Lmch
        Lmch = (maxFeat-minFeat)*x[j] + minFeat
    else:
        Lmch = max( Input_baseline[j] , minFeat)
        
    y[j] =Lmch

    j=5
    Lthroat_out = abs( (DmotiveOut/2-Dthroat/2)/(math.tan(math.radians(alphaMotiveDiff/2))) ) 
    Lconv= abs( (-Dthroat/2+DmotiveIn/2)/(math.tan(math.radians(alphaMotiveConv/2))) ) #min Lmotive
    Lmotive_min = (Lconv+Lthroat_out)

    minFeat = Lmotive_min*1.1
    maxFeat = Lmotive_min*2
    if featureDict["Lmotive"]==1: #Lmotive
        Lmotive = (maxFeat-minFeat)*x[j] + minFeat
    else:
        Lmotive = max( Input_baseline[j] , minFeat)
        # print(Input_baseline[j])
        # print(minFeat)
        # print(Lmotive)

        # if Lmotive < minFeat:
        #     Lmotive = minFeat*1.01
        #     print(Lmotive)

    y[j] =Lmotive
    # print("finally %f" %y[5])

    return y

