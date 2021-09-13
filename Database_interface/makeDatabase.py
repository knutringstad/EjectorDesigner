def makeDatabase():
    import pandas as pd
    import numpy as np
    import CurrentSettings.DatabaseSettings
    import CurrentSettings.SamplingSettings
    import CurrentSettings.BaselineConditions
    import CurrentSettings.CaseSettings
    from Sampling.sampling import sample
    import os

 

    params = CurrentSettings.DatabaseSettings.Parameters    

    N = CurrentSettings.SamplingSettings.NumberOfSamples
    featureDict = CurrentSettings.SamplingSettings.featureDict
    stdInput = CurrentSettings.BaselineConditions.InputStandard     # input from baseline conditions (if not a sampled feature)

    if CurrentSettings.DatabaseSettings.Database_from_testpointMatrix == False or not os.path.isfile('%s/TestPointsMatrix.csv' %CurrentSettings.CaseSettings.ResultsRoot):
        # If not using predefined testpoint matrix file or cannot find the testpoint matrix...
        # Make a testpoint matrix
        Databasepoints = sample(N,featureDict,stdInput,params)
        PointsDF = pd.DataFrame(data = Databasepoints, columns=params)
        PointsDF.to_csv('%s/TestPointsMatrix.csv' %CurrentSettings.CaseSettings.ResultsRoot,index=True)
    else:
        # else read that testpointfile
        PointsDF = pd.read_csv('%s/TestPointsMatrix.csv' %CurrentSettings.CaseSettings.ResultsRoot)


    # Make empty dataframe
    SimulationDF =  PointsDF.copy()
    SimulationDF[CurrentSettings.DatabaseSettings.Indicators + CurrentSettings.DatabaseSettings.Outputs] = np.nan

    filename = '%s/%s.csv' %(CurrentSettings.CaseSettings.ResultsRoot,CurrentSettings.DatabaseSettings.DatabaseName)
    
    # save the dataframe as the database

    if not os.path.isfile(filename): 
        SimulationDF.to_csv(filename, index=False)

    if SimulationDF[CurrentSettings.DatabaseSettings.Parameters].isna().any().any():
        raise Exception("Found NAN value in nessecary data for id %d (boundary conditions or geometry settings)" %id )