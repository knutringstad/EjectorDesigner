
        
def checkOutput(idnum,outputname):
    import CurrentSettings.CaseSettings, CurrentSettings.DatabaseSettings
    from EjectorDesignTool.efficiency import efficiencyPD
    import pandas as pd


    # READ DATAFRAME
    df = pd.read_csv('%s/%s.csv' %(CurrentSettings.CaseSettings.ResultsRoot,CurrentSettings.DatabaseSettings.DatabaseName))

    ResultsFolder = CurrentSettings.CaseSettings.ResultsRoot + '/DataFiles'

    # Go through the outputs in the results folder and updata in the dataframe

    if outputname in ['mfr_m','mfr_s','mfr_o','mfr_err','eff']:
        
        filename = ResultsFolder + '/MFR_HEM_EjectorML_ID_%d.dat' %idnum
        Simulated, Crashed, Convergence, mfr_m, mfr_s, mfr_o, mfr_err= readResultVariable('MFR',filename,idnum)

        df.loc[idnum,("SimulatedIndicator")] = Simulated
        df.loc[idnum,("CrashIndicator")] =Crashed
        df.loc[idnum,("ConvergenceIndicator")] =Convergence
        df.loc[idnum,("mfr_m")] =mfr_m
        df.loc[idnum,("mfr_s")] =mfr_s
        df.loc[idnum,("mfr_o")] =mfr_o
        df.loc[idnum,("mfr_err")] =mfr_err
        df.loc[idnum,("eff")] =efficiencyPD(df.iloc[idnum],'CO2')

    elif outputname == 'ds1':
        filename = ResultsFolder + '/entropy_integrals_%d.dat' %idnum
        entropy = readResultVariable('entropy',filename,idnum)
        df.loc[idnum,("ds1")] =entropy
     
    elif outputname == 'uni_alpha':
        filename = ResultsFolder + '/Uniformity_alpha_HEM_EjectorML_ID_%d.dat' %idnum
        uni_alpha = readResultVariable('uni_alpha',filename,idnum)
        df.loc[idnum,("uni_alpha")] = uni_alpha

    elif outputname == 'uni_vel':
        filename = ResultsFolder + '/Uniformity_alpha_HEM_EjectorML_ID_%d.dat' %idnum
        uni_vel = readResultVariable('uni_vel',filename,idnum)
        df.loc[idnum,("uni_vel")] =uni_vel


    else:
        raise Exception("Output %s has no defined output read mode! must be fixed" %outputname)
    
    

    df.loc[idnum,("Experiment_model_type")] ='HEM'
    df.loc[idnum,("MeshRefinement")] =0

    df.to_csv('%s/%s.csv' %(CurrentSettings.CaseSettings.ResultsRoot,CurrentSettings.DatabaseSettings.DatabaseName), index=False)








def readResultVariable(varName,filename,idnum):
    # Reads a file and postprocesses to the database
    # Takes in the variablename and the filename with the id of the simulation
    
    import CurrentSettings.CaseSettings
    import os
    import numpy as np

    if varName == 'MFR':
        Crashed = False
        Converged = False
        Simulated = True
        mfr_m = np.nan
        mfr_s = np.nan
        mfr_o = np.nan
        mfr_err = np.nan
        eff = np.nan

        if os.path.isfile(filename):
            datContent = [i.strip().split() for i in open(filename).readlines()]
            m = datContent[4]
            mfr_m = float(m[1])
            s = datContent[5]
            mfr_s = float(s[1])
            o = datContent[7]
            mfr_o = float(o[1])
            mfr_err = mfr_m+mfr_s+mfr_o
            if abs(mfr_err) < 5e-5:
                Converged = True

        else:
            
            filename = CurrentSettings.CaseSettings.ResultsRoot + '/FluentOutputs' + "/error_%d.dat" %idnum
            if os.path.isfile(filename):
                f = open(filename, "r")
                f.readline()
                errorMsg=f.read(1)
                if errorMsg == 'E':
                    Crashed = True
            else:
                Simulated = False
        
        return Simulated, Crashed, Converged, mfr_m, mfr_s, mfr_o, mfr_err

    ###################################################################################
    if varName == 'entropy':

        if os.path.isfile(filename): 
            datContent = [i.strip().split() for i in open(filename).readlines()]
            entropy =abs(float(datContent[8][1])-float(datContent[9][1]))/float(datContent[9][1]) 
            return entropy
        else:
            return np.nan

    ###################################################################################
    if varName == 'uni_alpha':

        if os.path.isfile(filename): 
            datContent = [i.strip().split() for i in open(filename).readlines()]
            uniformAlpha =datContent[5][1]
            return uniformAlpha
        else:
            return np.nan

    ###################################################################################
    if varName == 'uni_vel':

        if os.path.isfile(filename): 
            datContent = [i.strip().split() for i in open(filename).readlines()]
            uniformVel =datContent[5][1]
            return uniformVel
        else:
            return np.nan



