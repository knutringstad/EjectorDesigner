def journalFileHEM(ID,meshname,NIT):

    import pandas as pd
    import CurrentSettings.CFDSettings
    import CurrentSettings.CaseSettings
    import math
    import os


    # Read dataframe from CSV file 
    dfAll = pd.read_csv('%s/%s.csv' %(CurrentSettings.CaseSettings.ResultsRoot,CurrentSettings.DatabaseSettings.DatabaseName))
    df = dfAll.iloc[ID]

    #Get parameters from dataframe

    Pm =df['Pm']
    Ps =df['Ps']
    Po =df['Po']
    hm =df['hm']
    hs =df['hs']


    LintletConst = df['LintletConst']
    Lsuction = df['Lsuction']
    DmotiveIn = df['DmotiveIn']
    DmotiveOut = df['DmotiveOut']
    Dthroat = df['Dthroat']
    alphaMotiveDiff = df['alphaMotiveDiff']
    alphaMotiveConv = df['alphaMotiveConv']
    Lmch = df['Lmch']
    alphaSuction = df['alphaSuction']
    Dmix = df['Dmix']
    Lmix = df['Lmix']
    alphadiff = df['alphadiff']
    DdiffOut = df['DdiffOut']
    Loutlet = df['Loutlet']
    Dsuc = df['Dsuc']
    ThicknessNozzle = df['ThicknessNozzle']
    ThicknessNozzleOuter =df['ThicknessNozzleOuter']

    Ldiff = (DdiffOut/2-Dmix/2)/(math.tan(math.radians(alphadiff/2)))
    LinletDiff = (DmotiveOut/2-Dthroat/2)/(math.tan(math.radians(alphaMotiveDiff/2)))
    LinletConv = (-Dthroat/2+DmotiveIn/2)/(math.tan(math.radians(alphaMotiveConv/2)))

    Lmotive =LinletDiff + LinletConv
    alphaMotiveOuter = alphaSuction
    

    #Arbitrary temperatures (not used in calculations)
    Tm = 290
    Ts = 270
    To = 280

    #Setting up file structure

    MeshFolder = CurrentSettings.CaseSettings.MeshRoot +'/Meshes/'
    CaseFolder = CurrentSettings.CaseSettings.MeshRoot +'/Cases/'

    ResultsFolder = CurrentSettings.CaseSettings.ResultsRoot + '/DataFiles'

    Filename = CurrentSettings.CaseSettings.MeshRoot + '/JournalFiles/' + 'Journal_HEM_EjectorML_ID_%d.jou' % (ID)

    DefaultCaseName = CurrentSettings.CFDSettings.CFDcaseName


    # JOURNAL FILE FOR FLUENT


    f=open(Filename,'w') 

    f.write('%%Journal file for Geometry number %d\n'%  1) 
    f.write('%%Inlet motive P =%f bar, T= %fK \n'% ( Pm/100000,Tm)) 
    f.write('%%Inlet suction P =%f bar, T= %fK \n'% ( Ps/100000,Ts)) 

    if os.path.isfile('./%s/%s.cas' % (CaseFolder,meshname)):
        f.write('/file/read-case "%s/%s.cas" \n' % (CaseFolder,meshname)) 

    else:
        f.write('/file/read-case "%s"\n'  %DefaultCaseName )  
        f.write('/mesh/replace "%s/%s.msh" \n' % (MeshFolder,meshname))
        f.write('/file/write-case "%s/%s.cas" \n' % (CaseFolder,meshname))
    
    f.write('/define/user-defined/compiled compile "libudf_HRM" yes "UDS.c" "Properties_HEM_HRM.c" "" "" \n') 
    f.write('/define/user-defined/compiled load "libudf_HRM" \n') 
    f.write('/define/user-defined/execute-on-demand "read_input::libudf_HRM" \n') 
    f.write('/define/boundary-conditions pressure-inlet inlet_m yes no %d no 0 no %f no yes no no yes 5 1 no yes no yes no %d no 0\n'% (Pm,Tm,hm)) 
    f.write('/define/boundary-conditions pressure-out outlet yes no %d no %f no yes no no yes 5 1000 yes yes no 0 no 0 yes no no\n'% (Po,To)) 
    f.write('/define/boundary-conditions pressure-inlet inlet_s yes no %d no 0 no %f no yes no no yes 0.1 0.1 no yes no yes no %d no 1\n'% (Ps,Ts,hs)) 


    f.write('/solve/initialize/set-defaults/x-velocity 10 \n') 
    f.write('/solve/initialize/set-defaults/k 0.004 \n') 
    f.write('/solve/initialize/set-defaults/omega 100 \n') 
    f.write('/solve/initialize/set-defaults/uds-0 %d\n'% ((hs))) 
    f.write('/solve/initialize/set-defaults/temperature %f\n'%  Ts) 
    f.write('/solve/initialize/set-defaults/pressure %d\n'% (Pm*0.9) )

    f.write('/solve/set/discretization-scheme/k 0 \n') 
    f.write('/solve/set/discretization-scheme/omega 0 \n') 
    f.write('/solve/set/discretization-scheme/uds-0 1 \n') 
    f.write('/solve/set/discretization-scheme/temperature 1 \n') 
    f.write('/solve/set/discretization-scheme/pressure 14 \n') 
    f.write('/solve/set/discretization-scheme/mom 0 \n') 
    f.write('/solve/set/discretization-scheme/density 0 \n') 
    f.write('/solve/initialize/initialize-flow o \n') 
    f.write('/file/set-batch-options yes yes no\n ')



    f.write('/solve/iterate %d\n'% (math.ceil(NIT/2)) )   #first order
    f.write('/solve/set/p-v-controls %f %f %f\n' %(CurrentSettings.CFDSettings.cfl, CurrentSettings.CFDSettings.urelax, CurrentSettings.CFDSettings.prelax))  #first order 
    f.write('/solve/iterate %d\n'% (math.ceil(NIT/2)) )  #first order

    f.write('/solve/set/discretization-scheme/mom 1 \n') 
    f.write('/solve/set/discretization-scheme/density 1 \n') 
    f.write('/solve/set/discretization-scheme/epsilon 1 \n') 
    f.write('/solve/set/discretization-scheme/k 1 \n') 

    f.write('/solve/set/p-v-controls %f %f %f\n' %(CurrentSettings.CFDSettings.cfl, CurrentSettings.CFDSettings.urelax, CurrentSettings.CFDSettings.prelax))   #first order
    f.write('/solve/iterate %d\n'% (math.ceil(NIT/2)) )  
    f.write('/solve/set/p-v-controls %f %f %f\n' %(CurrentSettings.CFDSettings.cfl2, CurrentSettings.CFDSettings.urelax2, CurrentSettings.CFDSettings.prelax2))   
    f.write('/solve/iterate %d\n'% (math.ceil(NIT/2)) )   #second order


    # WRITE RESULTS


    if os.path.isfile('%s/Results_HEM_EjectorML_ID_%d.dat' %(ResultsFolder,ID)):
        f.write('/file/write-data "%s/Results_HEM_EjectorML_ID_%d.dat" ok\n'% (ResultsFolder,ID)) 
    else:
        f.write('/file/write-data "%s/Results_HEM_EjectorML_ID_%d.dat"\n'% (ResultsFolder,ID)) 

    if os.path.isfile('%s/MFR_HEM_EjectorML_ID_%d.dat'% (ResultsFolder,ID)) :
        f.write('/report/fluxes/mass-flow yes yes "%s/MFR_HEM_EjectorML_ID_%d.dat" \n'% (ResultsFolder,ID)) 
        f.write('no ok\n')
    else:
        f.write('/report/fluxes/mass-flow yes yes "%s/MFR_HEM_EjectorML_ID_%d.dat" \n'% (ResultsFolder,ID)) 

    f.write('/file/set-batch-options yes no no\n')


    # Definitions for post-processing KPI outputs

    #Motive In
    x0 = -Lmotive-LintletConst
    y0 =0
    x1 = -Lmotive-LintletConst
    y1 = DmotiveIn/2

    f.write('/surface/line-surface line-mot-in %f %f %f %f \n' %(x0, y0, x1, y1) ) 

    #Motive Throat

    x0 = -LinletDiff
    y0 =0
    x1 = -LinletDiff
    y1 = Dthroat/2

    f.write('/surface/line-surface line-mot-thr %f %f %f %f \n' %(x0, y0, x1, y1) ) 


    #Motive Out
    x0 = 0
    y0 =0
    x1 = 0
    y1 = DmotiveOut/2
    
    f.write('/surface/line-surface line-mot-out %f %f %f %f \n' %(x0, y0, x1, y1) ) 
    
    #Lmch In
    x0 = 0
    y0 =0
    x1 = 0
    y1 = Dmix/2 + Lmch*math.tan(math.radians(alphaSuction/2))
    
    f.write('/surface/line-surface line-lmch-in %f %f %f %f \n' %(x0, y0, x1, y1) ) 

    #Mixing In
    x0 = Lmch
    y0 =0
    x1 = Lmch
    y1 = Dmix/2
    
    f.write('/surface/line-surface line-mix-in %f %f %f %f \n' %(x0, y0, x1, y1) ) 

    #Mixing out
    x0 = Lmch+Lmix
    y0 =0
    x1 = Lmch+Lmix
    y1 = Dmix/2
    
    f.write('/surface/line-surface line-mix-out %f %f %f %f \n' %(x0, y0, x1, y1) ) 

    #Diff out
    x0 = Lmch+Lmix+Ldiff
    y0 =0
    x1 = Lmch+Lmix+Ldiff
    y1 = DdiffOut/2
    
    f.write('/surface/line-surface line-diff-out %f %f %f %f \n' %(x0, y0, x1, y1) ) 
# 'outlet', '1.9000959'], ['inlet_m', '1.3299177'], ['inlet_s', '1.8948318'], ['line-diff-out', '1.9253489'], ['line-lmch-in', '1.8563566'], ['line-mix-in', '1.8935942'], ['line-mix-out', '1.9255019'], ['line-mot-in', '1.3299177'], ['line-mot-out', '1.80041'], ['line-mot-thr', '1.3491757']

    if os.path.isfile('%s/entropy_integrals_%d.dat' %(ResultsFolder,ID)): 
        f.write('/report/surface-integrals/mass-weighted-avg "outlet" "inlet_m" "inlet_s" "line-diff-out" "line-lmch-in" "line-mix-in" "line-mix-out" "line-mot-in" "line-mot-out" "line-mot-thr" () udm-9 yes "%s/entropy_integrals_%d.dat" \n'  %(ResultsFolder,ID))
        f.write('no ok\n')
    else:
        f.write('/report/surface-integrals/mass-weighted-avg "outlet" "inlet_m" "inlet_s" "line-diff-out" "line-lmch-in" "line-mix-in" "line-mix-out" "line-mot-in" "line-mot-out" "line-mot-thr" () udm-9 yes "%s/entropy_integrals_%d.dat" \n'  %(ResultsFolder,ID))

        
    if os.path.isfile('%s/Uniformity_alpha_HEM_EjectorML_ID_%d.dat' %(ResultsFolder,ID)): #Outlet is surface ID = 3, check surface/surface-list
        f.write('/report/surface-integrals/uniformity-index-area-weighted 3 () udm-6 yes  "%s/Uniformity_alpha_HEM_EjectorML_ID_%d.dat" \n '% (ResultsFolder,ID)) 
        f.write('no ok\n')
    else:
        f.write('/report/surface-integrals/uniformity-index-area-weighted 3 () udm-6 yes  "%s/Uniformity_alpha_HEM_EjectorML_ID_%d.dat" \n'% (ResultsFolder,ID)) 


    if os.path.isfile('%s/Uniformity_uvel_HEM_EjectorML_ID_%d.dat' %(ResultsFolder,ID)): #Outlet is surface ID = 3, check surface/surface-list
        f.write('/report/surface-integrals/uniformity-index-area-weighted 3 () axial-velocity yes  "%s/Uniformity_uvel_HEM_EjectorML_ID_%d.dat" \n'% (ResultsFolder,ID)) 
        f.write('no ok\n')
    else:
        f.write('/report/surface-integrals/uniformity-index-area-weighted 3 () axial-velocity yes  "%s/Uniformity_uvel_HEM_EjectorML_ID_%d.dat" \n'% (ResultsFolder,ID) ) 

    if os.path.isfile('%s/motiveOutletPressure_avg_%d.dat' %(ResultsFolder,ID)): 
        f.write('/report/surface-integrals/mass-weighted-avg "line-mot-out" () udm-9 yes "%s/motiveOutletPressure_avg_%d.dat" \n'  %(ResultsFolder,ID))
        f.write('no ok\n')
    else:
        f.write('/report/surface-integrals/mass-weighted-avg  "line-mot-out" () pressure yes "%s/motiveOutletPressure_avg_%d.dat" \n'  %(ResultsFolder,ID))


    f.write('exit yes')

    f.close() 

    return Filename






