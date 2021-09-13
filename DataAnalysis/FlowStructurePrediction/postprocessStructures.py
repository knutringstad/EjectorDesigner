
from pickle import FALSE
from matplotlib.pyplot import scatter
from numpy.core.fromnumeric import var


def postprocessJournalFileHEM(ID,meshname):

    import pandas as pd
    import CurrentSettings.CFDSettings
    import CurrentSettings.CaseSettings
    import os
    import CurrentSettings.DatabaseSettings
    import CurrentSettings.CaseSettings
    import CurrentSettings.CFDSettings

    CaseFolder = CurrentSettings.CaseSettings.MeshRoot +'/Cases/'
    ResultsFolder = CurrentSettings.CaseSettings.ResultsRoot + '/DataFiles'
    Filename = CurrentSettings.CaseSettings.MeshRoot + '/JournalFiles/' + 'PostprocessingJournal_HEM_EjectorML_ID_%d.jou' % (ID)
    ResultsName = "Results_HEM_EjectorML_ID_%d" %ID

    f=open(Filename,'w') 

    printname= "pathline_plot_%d" %ID

    if os.path.isfile('./%s/%s.cas' % (CaseFolder,meshname)):
        f.write('/file/read-case "%s/%s.cas" \n' % (CaseFolder,meshname))
    else:
        return 

    f.write('/define/user-defined/compiled compile "libudf_HRM" yes "UDS.c" "Properties_HEM_HRM.c" "" "" \n') 
    f.write('/define/user-defined/compiled load "libudf_HRM" \n') 
    f.write('/define/user-defined/execute-on-demand "read_input::libudf_HRM" \n') 

    f.write('/file/read-data "%s/%s.dat" yes \n' % (ResultsFolder,ResultsName))
    
    f.write('/display/set/path-lines/maximum-steps 7000 \n')
    f.write('/display/path-lines/plot-write-xy-plot velocity-magnitude length 4 () 2 0 300 yes "%s/%s_motive_velocity"\n' %(ResultsFolder,printname))
    f.write('/display/path-lines/plot-write-xy-plot pressure length 4 () 2 600000 10000000 yes "%s/%s_motive_pressure"\n' %(ResultsFolder,printname))
    f.write('/display/path-lines/plot-write-xy-plot velocity-magnitude length 5 () 2 0 300 yes "%s/%s_suction_velocity"\n' %(ResultsFolder,printname))
    f.write('/display/path-lines/plot-write-xy-plot pressure length 5 () 2 600000 10000000 yes "%s/%s_suction_pressure"\n' %(ResultsFolder,printname))

    f.write('/plot/plot yes "%s/Axis_pressure_distribution_%d" yes no no pressure yes 1 0 1 () \n' % (ResultsFolder,ID) )
    f.write('/plot/plot yes "%s/Axis_velocity_distribution_%d" yes no no velocity-magnitude yes 1 0 1 () \n'% (ResultsFolder,ID))
    f.write('exit yes')
    f.close()
    return Filename


def runFluentPostprocess(JournalFileName,id):
    import subprocess

    # run fluent with journal file
    import CurrentSettings.DatabaseSettings
    import CurrentSettings.CaseSettings
    import CurrentSettings.CFDSettings

    outputsFolder = CurrentSettings.CaseSettings.ResultsRoot + '/FluentOutputs'
    outputfilename= "%s/output_post_%d.dat"%(outputsFolder,id)
    outputfile = open(outputfilename,"w")
    errorfilename ="%s/error_post_%d.dat"%(outputsFolder,id)
    errorfile = open(errorfilename,"w")
    
    command ='fluent -g 2ddp -t%d -i %s' %(CurrentSettings.CFDSettings.NumberCoresPerParallel , JournalFileName) 
    subprocess.call(command, stdout=outputfile,stderr=errorfile)
    outputfile.close()



def postprocessPathlines(filename,Average_over_N):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    p = -1 #particle counter
    X_p_all = []
    var_p_all=[]

    k=0

    if os.path.isfile(filename): 
        f = open(filename, "r")
        lines = f.readlines()

        for line in lines:
            
            if line.startswith("((xy/key/label \"particle-"):
                p=p+1        
                X=[]
                var=[]

            if line.startswith(")"):
                X= np.mean(np.array(X[:k-k%Average_over_N]).reshape(-1,Average_over_N),axis=1)
                X_p_all.append(X)
                var= np.mean(np.array(var[:k-k%Average_over_N]).reshape(-1,Average_over_N),axis=1)
                var_p_all.append(var)
                k=0


            if not line.startswith("(") and not line.startswith(")") and line.strip():
                data=line.strip().split()
                X.append(float(data[0]))
                var.append(float(data[1]))
                k=k+1


        # print(X_p_all)
        # print(X_p_all[0])

        # plt.plot(X_p_all[0],var_p_all[0])
        # plt.show()


        f.close() 

    return X_p_all, var_p_all



def iterate_over_pathlines_to_csv(N_average_over):
    
    import numpy as np
    import pandas as pd
    import CurrentSettings.CaseSettings

    resultsfolder= CurrentSettings.CaseSettings.ResultsRoot + '/DataFiles'
    d_mix_list = [0.003,0.0035,0.004,0.0045,0.005]
    l_mix_list = [0.01,0.02,0.03,0.04,0.05]
    variable_list=["pressure","velocity"]
    nozzles=["motive","suction"]

    for variable in variable_list:

        for nozzle in nozzles:
            id=0

            list=[]

            for l in l_mix_list:
                for d in d_mix_list:
                

                    filename = "%s/pathline_plot_%d_%s_%s" %(resultsfolder,id,nozzle,variable)
                    x_p_all, var_p_all = postprocessPathlines(filename,N_average_over) 

                    storage_all_p = []

                    
                    for p in range(len(x_p_all)):
                        L = len(var_p_all[p])
                        storagearray_p =np.ones( (L,4) )

                        storagearray_p[:,0] = storagearray_p[:,0]*d
                        storagearray_p[:,1] = storagearray_p[:,1]*l
                        storagearray_p[:,2] = x_p_all[p]
                        storagearray_p[:,3] = var_p_all[p]

                        storage_all_p.append(storagearray_p)

                        df=pd.DataFrame(storagearray_p,columns=["Dmix","Lmix","X",variable])
                        df.to_csv("DataAnalysis/FlowStructurePrediction/PathData/%s_%s_particle%d_pathlines_id_%d.csv" %(variable,nozzle,p,id))

                    id=id+1
                    # list.append(storage_all_p)








def concatinate_dataframes(variable,nozzle,particle):
    import pandas as pd
    import os
    idmax=24
    frames=[]
    for id in range(idmax):
        filename="DataAnalysis/FlowStructurePrediction/PathData/%s_%s_particle%d_pathlines_id_%d.csv" %(variable,nozzle,particle,id)
        if os.path.isfile(filename ):
            df = pd.read_csv(filename,index_col=0)
            frames.append(df)

    result = pd.concat(frames,ignore_index=True)
    result.to_csv("DataAnalysis/FlowStructurePrediction/PathDataAnalysed/%s_%s_particle%d_pathlines.csv" %(variable,nozzle,particle))





def GPRflowstructure():
    from joblib import dump, load
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel ,RationalQuadratic,ExpSineSquared
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    from matplotlib import cm
    
    
    
    folder="DataAnalysis/FlowStructurePrediction/PathDataAnalysed"
    variable="velocity"
    nozzle="suction"
    particle_list=[2]
    for particle in particle_list:


        filename="%s/%s_%s_particle%d_pathlines.csv" %(folder,variable,nozzle,particle)
        df = pd.read_csv(filename,index_col=0)

        features=['Dmix','Lmix','X']
        output = variable


        df_2=df
        
        predictionTest=False
        if predictionTest:
            df=df.loc[ (df["Dmix"]!=0.004) | (df["Lmix"]!=0.03) ]




        trainBool=True
        if trainBool == True:
            sample_size =0.1
            seed = 2

            x_train, x_test, y_train, y_test = train_test_split(df[features],df[output], test_size=sample_size,random_state=seed)

            x_train = np.array(x_train)
            x_test = np.array(x_test)

            # Scaling
            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)

            lengthScale = np.ones(len(features))*0.5
            std_estimate= 0.000

            kernel = ConstantKernel(1e+2) * RBF(length_scale=lengthScale)  + WhiteKernel(noise_level=1e-2) 

            Normalize_bool=True

            gp = GaussianProcessRegressor(kernel=kernel,alpha=std_estimate**2,n_restarts_optimizer=2,normalize_y=Normalize_bool).fit(x_train, y_train)  #run fitting

            gp_dump = dump(gp,'gp_flow.joblib')
            sc_dump = dump(sc,'sc_flow.joblib')
            x_test_dump = dump(x_test,'x_test_flow.joblib')
            x_train_dump = dump(x_train,'x_train_flow.joblib')
            y_test_dump = dump(y_test,'y_test_flow.joblib')
            y_train_dump = dump(y_train,'y_train_flow.joblib')
            print("Model trained")
        else:
            print("Loading pretrained gaussian process")
            gp= load('gp_flow.joblib') 
            sc = load('sc_flow.joblib')
            x_test = load('x_test_flow.joblib')
            x_train = load('x_train_flow.joblib')
            y_test = load('y_test_flow.joblib')
            y_train = load('y_train_flow.joblib')




        kern = gp.kernel_
        print(kern)
        pred = gp.predict(x_test)
        err=mean_absolute_error(y_test,pred)
        err_square=mean_squared_error(y_test,pred)

        print(err)

        plottype="singlevar"
        if plottype =="surf":

            var1=0
            var2=2



            Ndim = len(features)        

            N_points=40
            scaled_minmax=1.6
            x_vec = np.linspace(-scaled_minmax, scaled_minmax, N_points)
            y_vec = np.linspace(-scaled_minmax, scaled_minmax, N_points)

            mat = np.meshgrid(x_vec, y_vec)
            X_,Y_ = np.meshgrid(x_vec, y_vec)

            numPoints = len(mat[0].flatten('C'))
            testPoints=np.zeros((Ndim,numPoints))


            
            if var1==0:
                freevarval=0.04 
                freevar= np.ones(numPoints)*freevarval  #SET THE FREE VAR HERE
            else:
                freevarval=0.003
                freevar= np.ones(numPoints)*freevarval

            vecunscaled = np.array([freevar,freevar,freevar])
            vecscaled=sc.transform(vecunscaled.T).T

            for j in range((Ndim)):
                if j == var1:
                    testPoints[j]=mat[0].flatten('C')
                elif j==var2:
                    testPoints[j]=mat[1].flatten('C')
                else:
                    testPoints[j] = vecscaled[j]






            scaled=np.array([x_vec,x_vec,x_vec])
            unscaled=sc.inverse_transform(scaled.T)


            X_,Y_ = np.meshgrid(unscaled[:,var1], unscaled[:,var2])



            pred_grid, std_grid = gp.predict(testPoints.T, return_std=True)

            pred_grid= np.reshape(pred_grid, (N_points,N_points),order='C')
            std_grid= np.reshape(std_grid, (N_points,N_points),order='C')


            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            # ax.set_zscale('log')

            surf = ax.plot_surface(X_,Y_, pred_grid[:,:],alpha=0.6)

            plt.tight_layout()

            # surf = ax.plot_surface(X_,Y_, pred_grid[:,:] )

            
            if (not var1==1 and not var2==1):
                scatter_df=df_2.loc[ (df_2["Lmix"]== freevarval) ]
            else:
                scatter_df=df_2.loc[ (df_2["Dmix"]== freevarval) ]

            plotline3d=True
            if plotline3d:
                first= [0,38,38+43,38+45+45+1,38+45+45+41+13]
                last = [38,38+43,38+45+45+1,38+45+45+41+13,38+45+45+45+45+22]
                for i in range(5):
                    ax.plot3D(scatter_df[features[var1]][first[i]:last[i]], scatter_df[features[var2]][first[i]:last[i]], scatter_df[variable][first[i]:last[i]], 'red')

            ax.scatter(scatter_df[features[var1]],scatter_df[features[var2]], scatter_df[variable], c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
           

            ax.set_ylabel('Pathline length - c [m]')
            ax.set_xlabel(r'Mixing diameter - $D_{{mix}}$ [m]')
            ax.set_zlabel(r'Velocity magnitude - $\|\vec{u}\|$ [m/s]')
            ax.set_zlim(0, 80)
            ax.set_ylim(0.0, 0.13)


            
        elif plottype=="singlevar":

            Ndim = len(features)        

            N_points=40
            x_vec = np.linspace(0, 0.13, N_points)

            dmix=0.004
            lmix=0.03
            # mix_list=[0.003,0.0035,0.004,0.0045,0.005]
            mix_list=[0.01,0.02,0.03,0.04,0.05]
            # mix_list=[dmix]
            for lmix in mix_list:
                

                dmixvec= np.ones(N_points)*dmix 
                lmixvec= np.ones(N_points)*lmix

                vecunscaled = np.array([dmixvec,lmixvec,x_vec])
                vecscaled=sc.transform(vecunscaled.T).T

                testPoints=np.zeros((Ndim,N_points))

                testPoints[0] = vecscaled[0]
                testPoints[1] = vecscaled[1]
                testPoints[2] = vecscaled[2]

                pred, std= gp.predict(testPoints.T, return_std=True)
                
                scatter_df=df_2.loc[ (df_2["Dmix"]== dmix) ]
                scatter_df=scatter_df.loc[ (scatter_df["Lmix"]== lmix) ]


                if variable=="pressure": #scale to bar
                    y_scatter=np.divide(scatter_df[variable],1e+5)
                    pred=np.divide(pred,1e+5)
                    std=np.divide(std,1e+5)
                    ymin = 29
                    ypos_label=ymin+0.3
                    y_ax=ymin
                elif variable=="velocity": #scale
                    y_scatter=np.divide(scatter_df[variable],1)
                    pred=np.divide(pred,1)
                    std=np.divide(std,1)
                    ypos_label=3
                    y_ax=0

                plt.plot(x_vec,pred)
                # plt.fill_between(x_vec, pred - std, pred + std, color='darkblue', alpha=0.2)
                plt.fill_between(x_vec, pred - std, pred + std,  alpha=0.2)


                # plt.scatter(scatter_df["X"],y_scatter)



                if variable=="pressure":
                    plt.ylabel("Pressure [bar]")
                    plt.ylim(ymin,39)
                elif variable=="velocity":
                    plt.ylabel(r"Velocity magnitude - $\| \vec{u} \|$ [m/s]")
                    plt.ylim(0,100)

                
                plt.annotate('', xy=(0.0, y_ax), xytext=(0.015, y_ax), arrowprops=dict(arrowstyle='<->',facecolor='black'), annotation_clip=False            )
                plt.annotate('Suction nozzle', xy=(0.0, ypos_label), xytext=(-0.002, ypos_label), annotation_clip=False            )
                plt.annotate('', xy=(0.02, y_ax), xytext=(0.06, y_ax), arrowprops=dict(arrowstyle='<->',facecolor='black'), annotation_clip=False            )
                plt.annotate('Mixing chamber', xy=(0.026, ypos_label), xytext=(0.03, ypos_label), annotation_clip=False            )
                plt.annotate('', xy=(0.06, y_ax), xytext=(0.125, y_ax), arrowprops=dict(arrowstyle='<->',facecolor='black'), annotation_clip=False            )
                plt.annotate('Diffuser', xy=(0.075, ypos_label), xytext=(0.073, ypos_label), annotation_clip=False            )
                
                plt.tight_layout

                plt.xlabel("Pathline distance - c [m]")

                
    # legends= [r'$D_{{mix}}$=3 [mm], $\eta=0.21$ [-]',r'$D_{{mix}}$=3.5 [mm], $\eta=0.32$ [-]',r'$D_{{mix}}$=4 [mm], $\eta=0.34$ [-]',r'$D_{{mix}}$=4.5 [mm], $\eta=0.25$ [-]',r'$D_{{mix}}$=5 [mm], $\eta=0.05$ [-]']
    legends= [r'$L_{{mix}}$=10 [mm], $\eta=0.40$ [-]',r'$L_{{mix}}$=20 [mm], $\eta=0.42$ [-]',r'$L_{{mix}}$=30 [mm], $\eta=0.37$ [-]',r'$L_{{mix}}$=40 [mm], $\eta=0.34$ [-]',r'$L_{{mix}}$=50 [mm], $\eta=0.30$ [-]']
    plt.legend(legends)
    plt.show()




def concate_all():


    variable_list=["pressure","velocity"]
    nozzles=["motive","suction"]
    particle_list=[0,1,2,3,4]

    for variable in variable_list:
        for nozzle in nozzles:
            for particle in particle_list:
                concatinate_dataframes(variable,nozzle,particle)




def startPost():



    # for i in range(25):
    #     journal=postprocessJournalFileHEM(i,"ML_structure_prediction_%d.000000_2D" %i)
    #     runFluentPostprocess(journal,i)
    #     print(i)


    # iterate_over_pathlines_to_csv(30)

    # concate_all()

    GPRflowstructure()
