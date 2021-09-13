def updateDataBase():
    import pandas as pd
    import numpy as np
    import CurrentSettings.DatabaseSettings, CurrentSettings.CaseSettings

    df = pd.read_csv('%s/%s.csv' %(CurrentSettings.CaseSettings.ResultsRoot,CurrentSettings.DatabaseSettings.DatabaseName))

    simulated  = df.loc[ (df["SimulatedIndicator"]==True)].index
    updateResults(list(simulated))

    # Go through all datapoints with SimulatedIndicator == False or NaN
    # Simulate 
    # Post process, read into dataframe

    print('Updating new datapoints...')
    ids = df.index[ ( df['SimulatedIndicator'].isnull() ) | (df['SimulatedIndicator']==False )  ].tolist()

    flags = [False, False, 0] #Continue_flag, relaxflag, refineflag (level of iteration refinement CFD)

    simulate(ids,flags)

    print('Done updating new datapoints...')

    # Go through all datapoints with SimulatedIndicator == True and ConvergenceIndicator == False and CrashIndicator == False
    # Simulate
    # Post process, read into dataframe (only if not crashed and result has improved)

    # Go through all SimulatedIndicator == True and CrashIndicator == True
    # Simulate 
    # Post process, read into dataframe 

    
    print("To be made")






def simulate(ids,flags):
    import CurrentSettings.CFDSettings
    import math

    cleanFolder()

    if CurrentSettings.CFDSettings.SimulationMode == 'local':
        print('Simulating locally...')

        # Divides the datapoint tasks among the paralell scripts

        numberPar= CurrentSettings.CFDSettings.NumberParallel #number of paralell script threads
        maxIterationsPar = math.floor(len(ids)/numberPar )    # number of iterations to simulate all datapoints with numberPar parallel threads 


        for j in range(maxIterationsPar): # Go through the rounds of parallel thread iterations

            idsThread = ids [j*numberPar:j*numberPar+numberPar]   # the ids that are calculated in 
            RunParallelLocal(idsThread,numberPar,flags)
            
        lastrunsid = maxIterationsPar*numberPar
        lastid = len(ids)
        N = lastid - lastrunsid

        #calculation of the remainder of datapoints

        if N>0:
            idsThread = ids [lastrunsid:lastid]
            RunParallelLocal(idsThread,N,flags)

        print('Completed simulations')



    else:
        raise Exception ("SimulationMode not found or not implemented yet")








def RunParallelLocal(idsThread,numberPar,flags):     
    import threading
    import CurrentSettings.CFDSettings
    import time

    print('Starting parallel threads with ids:')
    print(idsThread)

    # starting threads for CFD calculations and setup

    threads = []
    for i in range(numberPar):
        t = threading.Thread(target=runSimulation, args=(idsThread[i], flags ))
        threads.append(t)
        t.start()
        time.sleep(30)

    for x in threads:
        x.join()    

    updateResults(idsThread)




def runSimulation(id,flags):
    
    #id == simulation ID to be run
    #flags = [ continue flag (open a previously run simulation), relaxflag (decrease underrelaxation), refineflag (level of iterations)]
    import csv
    import os
    import CurrentSettings.DatabaseSettings
    import CurrentSettings.CaseSettings
    import CurrentSettings.CFDSettings
    import pandas as pd
    
    cleanFolder()

    nit = setNumberOfIterations(flags[2])
    # setRelaxationSettings(flags[1])

    MeshName = MakeMesh(id)

    ModelSetup(id)

    JournalFileName = MakeJournalFile(id,MeshName,nit)



    print('Running simulation id: %d, name : %s, NIT=%d' %(id,MeshName,nit))

    # run fluent with journal file
    import subprocess
    outputsFolder = CurrentSettings.CaseSettings.ResultsRoot + '/FluentOutputs'
    outputfilename= "%s/output_%d.dat"%(outputsFolder,id)
    outputfile = open(outputfilename,"w")
    errorfilename ="%s/error_%d.dat"%(outputsFolder,id)
    errorfile = open(errorfilename,"w")
    
    command ='fluent -g 2ddp -t%d -i %s' %(CurrentSettings.CFDSettings.NumberCoresPerParallel , JournalFileName) 
    subprocess.call(command, stdout=outputfile,stderr=errorfile)
    outputfile.close()
    
    print('Completed simulation id: %d, name : %s, NIT=%d' %(id,MeshName,nit))



def updateResults(idsThread):
    import CurrentSettings.CaseSettings, CurrentSettings.DatabaseSettings
    from Database_interface.checkResults import checkOutput
    import pandas as pd

    print('Updating results...')

    for idnum in idsThread:
        #check results folder
        for output in CurrentSettings.DatabaseSettings.Outputs:
            checkOutput(idnum, output)
        





def setNumberOfIterations(refineflag):
    nit = 40000 #number of iterations
    if refineflag ==1:
        nit=70000
    if refineflag ==2:
        nit=100

    return nit


def MakeMesh(id):
    import CurrentSettings.DatabaseSettings
    import CurrentSettings.CFDSettings
    import CurrentSettings.MeshingSettings
    import pandas as pd
    import os.path

    df = pd.read_csv('%s/%s.csv' %(CurrentSettings.CaseSettings.ResultsRoot,CurrentSettings.DatabaseSettings.DatabaseName))
    idData = df.iloc[id]

    meshID = CurrentSettings.CaseSettings.CurrentCaseName + '_'
    data = '%f' %id
    meshID = meshID + data
        

    if CurrentSettings.CFDSettings.dimensions == '2D':    
        from CFDinterface.meshing import meshingEjector

        meshID = meshID + '_2D'

        write_mesh =1  #Should RPL file also make and save mesh?
        run_ICEM = 1  #Run ICEM in script?
        mesh_smoothing=0  #Smoothing the mesh after generation. WARNING: TAKES A LONG TIME
        mesh_convergence =0  #Mesh refined with mesh scale 'refine'
        refine = 1  #  0.25, 0.5, 2, 3
        delta = CurrentSettings.MeshingSettings.MeshSize  #baseline delta
        if CurrentSettings.MeshingSettings.meshingMode == "Const_mix_number":
            delta = idData["Dmix"]/CurrentSettings.MeshingSettings.Ncells_mix

        MeshRootLocation = CurrentSettings.CaseSettings.MeshRoot + '/Meshes'
        ScriptRootLocation = CurrentSettings.CaseSettings.MeshRoot + '/MeshData'

        if not os.path.isfile( '%s/%s.msh' %(MeshRootLocation,meshID)):
            meshingEjector(MeshRootLocation, ScriptRootLocation,meshID,run_ICEM, write_mesh, mesh_convergence,refine,mesh_smoothing, delta, idData)

    else:
        raise Exception('only 2D is implemented so far...')
    return meshID







def MakeJournalFile(id,MeshName,NIT):
    from CFDinterface.makeJournal import makeJournal

    # check component type, check continue and so on
    return makeJournal(id,MeshName,NIT)






def ModelSetup(id):
    import CurrentSettings.CFDSettings
    from CFDinterface.ModelSetup import DHRMsetup

    if CurrentSettings.CFDSettings.model =='DHRM':
        DHRMsetup(id)



def cleanFolder( ):
    # Removes files from previous Fluent simulations
    import os
    for file in os.listdir("./"):
        if file.endswith(".bat"):
            os.remove(file)
    for file in os.listdir("./"):
        if file.endswith(".log"):
            os.remove(file)


