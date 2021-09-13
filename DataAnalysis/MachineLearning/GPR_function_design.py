import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import csv
import statistics
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 
import CoolProp.CoolProp as CP
import matplotlib.tri as tri
from joblib import dump, load
import plotly.graph_objects as go

def GPR_function(df,features, output, sample_size, seed, displayBool,optimizeBool,trainBool):

    # Drop crashed simulations, clean up
    df = df.dropna()
    df = df.drop(df.loc[df["CrashIndicator"]==1].index)
    df = df.drop(df.loc[df["uni_vel"]==0].index)
    df = df.drop(df.loc[abs(df["mfr_err"])>0.0001].index)


    if output[0].startswith("ds_"):
        df = df.drop(df.loc[df["mfr_s"]<0].index)

    df_ejector = df.copy() 
    df_ejector["Plift"] = df_ejector["Po"]- df_ejector["Ps"]
    df_ejector["ER"] = df_ejector["mfr_s"] / df_ejector["mfr_m"]
    # df_ejector = df_ejector.drop(df.loc[df_ejector["ER"]>2.5].index)
    effv = df_ejector["ER"].copy()

    for index, row in df_ejector.iterrows():
        Pm =df_ejector["Pm"][index]
        Ps=df_ejector["Ps"][index]
        hs=  df_ejector["hs"][index]
        ER=df_ejector["ER"][index]
        hm=df_ejector["hm"][index]
        Po=df_ejector["Po"][index]
        effv[index]=efficiency(Pm, Po, Ps, hm ,hs, ER, "CO2")
        
        if effv[index]<0:
            effv[index]=0

    # print(effv)
    df_ejector["eff"] = effv


    
    #Gaussian Process regression
    #Load or retrain

    if trainBool == True:
        print("Training GPR model")
        x = df_ejector[features]
        y = df_ejector[output]

        # Split into training and testing set
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=sample_size,random_state=seed)

        # Scaling
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)


        lengthScale = np.ones(len(features))*0.5
        std_estimate= 0.0000
        kernel = 0.2 * RBF(length_scale=lengthScale, length_scale_bounds=(1e-4, 3e2))  + WhiteKernel(noise_level=1e-2)#* RBF(length_scale=lengthScale, length_scale_bounds=(2e5, 5e6)) #  +0.1 * RBF(length_scale=1E+3, length_scale_bounds=(1e3, 1e4))
        # kernel = 0.2 * RBF(length_scale=lengthScale, length_scale_bounds=(1e-3, 3e1))  + WhiteKernel(noise_level=1e-2)* RBF(length_scale=lengthScale, length_scale_bounds=(1e-3, 3e1))
        gp = GaussianProcessRegressor(kernel=kernel,alpha=std_estimate**2,n_restarts_optimizer=20).fit(x_train, y_train)  #run fitting
        gp_dump = dump(gp,'gp_geom.joblib')
        sc_dump = dump(sc,'sc_geom.joblib')
        x_test_dump = dump(x_test,'x_test_geom.joblib')
        x_train_dump = dump(x_train,'x_train_geom.joblib')
        y_test_dump = dump(y_test,'y_test_geom.joblib')
        y_train_dump = dump(y_train,'y_train_geom.joblib')
        print("Model trained")
    else:
        print("Loading pretrained gaussian process")
        gp= load('gp_geom.joblib') 
        sc = load('sc_geom.joblib')
        x_test = load('x_test_geom.joblib')
        x_train = load('x_train_geom.joblib')
        y_test = load('y_test_geom.joblib')
        y_train = load('y_train_geom.joblib')

    kern = gp.kernel_
    pred = gp.predict(x_test)
    err=mean_absolute_error(y_test,pred)
    err_square=mean_squared_error(y_test,pred)

    print("Avg abs error: %.9f "%err)
    print("Avg square error: %.9f "%err_square)


    if displayBool:
        scatterplot=False
        if scatterplot==True:
            #------------------SCATTER PLOT FIGURE -------------------------
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(x_test[:,1],x_test[:,4], pred, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
            ax.scatter(x_test[:,1],x_test[:,4], y_test, c='g', s=50, zorder=10, edgecolors=(0, 0, 0))

            ax.scatter(x_train[:,1],x_train[:,4], y_train, c='b', s=50, zorder=10, edgecolors=(0, 0, 0))
            ax.set_zlim([-0.5,0.5])

            # ax.set_xlabel("Mixing chamber diameter [-]")
            # ax.set_ylabel("Diffuser outlet diameter [-]")
            # ax.set_zlabel("mass conservation error [kg/s]")

        heatmapplot=True
        if heatmapplot==True:
            #------------------HEAT MAP FIGURE -------------------------
            var1 =1
            var2 = 2

            fig = plt.figure()

            Ndim = len(features)        

            N=100
            x_vec = np.linspace(-3, 3, N)
            y_vec = np.linspace(-3, 3, N)
            mat = np.meshgrid(x_vec, y_vec)
            numPoints = len(mat[0].flatten('C'))
            testPoints=np.ones((Ndim,numPoints))


            # d1_plot= np.ones(numPoints)*85e+5 #Pm
            # d2_plot= np.ones(numPoints)*45e+5 #Po
            # d3_plot= np.ones(numPoints)*35e+5 #Ps
            # d4_plot= np.ones(numPoints)*2.80e+5 #hm
            # d5_plot= np.ones(numPoints)*4.3e+5 #hs


            # d1_plot= np.ones(numPoints)*1.77e-03 #dmo / Lmch
            # d2_plot= np.ones(numPoints)*3e-03#Dmix
            # d3_plot= np.ones(numPoints)*1e-02 #Lmix
            # d4_plot= np.ones(numPoints)*11.6#alphadiff
            # d5_plot= np.ones(numPoints)*2.81e-02 #DdiffOut

            # d1_plot= np.ones(numPoints)*0.0001 #dmo / Lmch
            # d2_plot= np.ones(numPoints)*0.00257 #Dmix
            # d3_plot= np.ones(numPoints)*0.0234 #Lmix
            # d4_plot= np.ones(numPoints)*5#alphadiff
            # d5_plot= np.ones(numPoints)*0.06 #DdiffOut

            baselinedesign = [0.00152,0.004,0.026,6,0.012]

            # #Individual optimums
            d1_plot= np.ones(numPoints)*1.77e-03 #dmo / Lmch
            d2_plot= np.ones(numPoints)*3.98e-03#Dmix
            d3_plot= np.ones(numPoints)*2.3e-02 #Lmix
            d4_plot= np.ones(numPoints)*11.6#alphadiff
            d5_plot= np.ones(numPoints)*2.81e-02 #DdiffOut

            
            vecunscaled = np.array([d1_plot,d2_plot,d3_plot,d4_plot,d5_plot])
            vecscaled=sc.transform(vecunscaled.T).T



            # print(evaluate(vecscaled[:,0],gp))

            for j in range((Ndim)):
                if j == var1:
                    testPoints[j]=mat[0].flatten('C')
                elif j==var2:
                    testPoints[j]=mat[1].flatten('C')
                else:
                    testPoints[j] = vecscaled[j]



            pred_grid, std_grid =  gp.predict(testPoints.T, return_std=True)
            pred_grid= np.reshape(pred_grid, (N,N),order='C')
            std_grid = np.reshape(std_grid, (N,N),order='C')

            scaled=np.array([x_vec,x_vec,x_vec,x_vec,x_vec])
            unscaled=sc.inverse_transform(scaled.T)


            #------z axis --------
            cmin = 0.0
            cmax= 0.45
            ncolors=10
            levels =np.linspace(cmin,cmax,ncolors)


            diagram="design"
            if diagram =="ph":
                
                pressurebar = np.divide(unscaled[:,var2],10**5)
                enthalpyKjkg = np.divide(unscaled[:,var1],1000)
                plt.contourf(enthalpyKjkg,pressurebar,pred_grid[:,:],levels=levels)

                cbar=plt.colorbar(label='Efficiency [-], eqn. (1)')
                plt.clim(cmin,cmax)
                
                p_vec = np.arange(50e5, CP.PropsSI('PCRIT','CO2'), 1000)
                CO2_hsat_v_vec = np.zeros(len(p_vec))
                CO2_hsat_l_vec = np.zeros(len(p_vec))

                for i,p in enumerate(p_vec):
                    CO2_hsat_l_vec[i] = CP.PropsSI('H','P',p_vec[i],'Q',0,'CO2')/1000
                    CO2_hsat_v_vec[i] = CP.PropsSI('H','P',p_vec[i],'Q',1,'CO2')/1000
                
                p_vec=np.divide(p_vec,10**5)



                plt.plot(CO2_hsat_l_vec,p_vec,'k')
                plt.plot(CO2_hsat_v_vec,p_vec,'k')
                plt.plot([CO2_hsat_l_vec[-1],CO2_hsat_v_vec[-1]],[p_vec[-1],p_vec[-1]],'k')

                plt.xlim([2.72e+2,3.5e+2])
                plt.xticks(np.arange(2.72e+2,3.5e+2, step=25))

                plt.ylim([60,150])
                plt.yticks(np.arange(60,150, step=20))
                
                plt.xlabel("Enthalpy [kJ/kg]")
                plt.ylabel("Pressure [Bar]")

            elif diagram=="design":                              
                mm_y = np.multiply(unscaled[:,var2],1000)
                
                mm_x = np.multiply(unscaled[:,var1],1000)
                # mm_x = np.multiply(unscaled[:,var1],1)
                plt.contourf(mm_x,mm_y,pred_grid[:,:],levels=levels,extend='both')

                cbar=plt.colorbar(label='Efficiency [-], Eqn. (20)')
                plt.clim(cmin,cmax)
                
                plt.xlim([0,8])
                # plt.xlim([0,90])
                # plt.xticks(np.arange(2.72e+2,3.5e+2, step=25))

                plt.ylim([0,50])
                # plt.ylim([0,60])
                # plt.yticks(np.arange(60,150, step=20))

                LabelList = [r'Motive outlet diameter - D$_{{m-out}}$ [mm]',r'Mixing chamber diameter - D$_{{mix}}$ [mm]',r'Mixing chamber length - L$_{{mix}}$ [mm]',r' Diffuser angle - $\alpha_{d}$  [deg]',r'Diffuser diameter - D$_{{diff}}$  [mm]']

                plt.xlabel(LabelList[var1],fontsize=12)
                plt.ylabel(LabelList[var2],fontsize=12)
                plt.tight_layout()


        stdplot=False
        if stdplot==True:
            # STANDARD DEVIATION PLOT

            fig = plt.figure()

            cmin = 0
            cmax= 0.1
            levels =np.linspace(cmin,cmax,ncolors)

            plt.contourf(unscaled[:,var1],unscaled[:,var2],std_grid[:,:],levels=levels,extend='both')
            plt.xlim([max(min(unscaled[:,var1]),0),max(unscaled[:,var1])])
            plt.ylim( [max(min(unscaled[:,var2]),0),max(unscaled[:,var2])])

            if diagram =="ph":
                
                p_vec = np.arange(50e5, CP.PropsSI('PCRIT','CO2'), 1000)
                CO2_hsat_v_vec = np.zeros(len(p_vec))
                CO2_hsat_l_vec = np.zeros(len(p_vec))

                for i,p in enumerate(p_vec):
                    CO2_hsat_l_vec[i] = CP.PropsSI('H','P',p_vec[i],'Q',0,'CO2')
                    CO2_hsat_v_vec[i] = CP.PropsSI('H','P',p_vec[i],'Q',1,'CO2')
                
                
                plt.plot(CO2_hsat_l_vec,p_vec,'k')
                plt.plot(CO2_hsat_v_vec,p_vec,'k')
                plt.plot([CO2_hsat_l_vec[-1],CO2_hsat_v_vec[-1]],[p_vec[-1],p_vec[-1]],'k')
                plt.xlim([2.26e+5,3.7e+5])
                plt.xticks(np.arange(2.26e+5,3.7e+5, step=5e+4))
                
                plt.xlabel("Enthalpy [j/kg]")
                plt.ylabel("Pressure [Pa]")

            elif diagram=="design":
                LabelList = [r'Motive outlet diameter - D$_{{m-out}}$ [mm]',r'Mixing chamber diameter - D$_{{mix}}$ [mm]',r'Mixing chamber length - L$_{t{mix}}$ [mm]',r' Diffuser angle - $\alpha_{d}$  [deg]',r'Diffuser diameter - D$_{{diff}}$  [mm]']

                plt.xlabel(LabelList[var1],fontsize=12)
                plt.ylabel(LabelList[var2],fontsize=12)

            plt.colorbar()
            plt.clim(cmin,cmax)


        individualOptPlot=False
        if individualOptPlot:
            # IDIVIDUAL OPTIMUM




            fig,ax= plt.subplots()
            individualOptimal = np.ones(5)

            legend=[r"D$_{{m-out}}$",r"D$_{{mix}}$",r"L$_{{mix}}$",r"$\alpha_d$",r"D$_{{diff}}$"]
            colorlines=["red","green", "blue", "black", "purple"]

            for varX in range(5):
                N=50
                x_vec = np.linspace(-2, 2, N)
                
                numPoints = len(x_vec)
                testPoints=np.ones((Ndim,numPoints))

                d1_plot= np.ones(numPoints)*0.00152 #dmo
                d2_plot= np.ones(numPoints)*0.004 #Dmix
                d3_plot= np.ones(numPoints)*0.026 #Lmix
                d4_plot= np.ones(numPoints)*6 #alphadiff
                d5_plot= np.ones(numPoints)*0.012 #DdiffOut
                

                vecunscaled = np.array([d1_plot,d2_plot,d3_plot,d4_plot,d5_plot])
                vecscaled=sc.transform(vecunscaled.T).T

                for j in range((Ndim)):
                    if j == varX:
                        testPoints[j]=x_vec
                    else:
                        testPoints[j] = vecscaled[j]



                pred_line, std_line =  gp.predict(testPoints.T, return_std=True)

                scaled=np.array([x_vec,x_vec,x_vec,x_vec,x_vec])
                unscaled=sc.inverse_transform(scaled.T)

                dimlessX= np.divide(unscaled[:,varX],max(unscaled[:,varX]))
                
                ax.plot(dimlessX,pred_line,c=colorlines[varX],label=legend[varX])
                
                # ax.plot(dimlessX,pred_line+std_line[0],'--',c=colorlines[varX],label='_Hidden label')
                # ax.plot(dimlessX,pred_line-std_line[0],'--',c=colorlines[varX],label='_Hidden label')
                
                ax.set_ylim([0,0.5])
                ax.set_xlim([0,1])

                indexMax= np.argmax(pred_line)
                individualOptimal[varX]= unscaled[indexMax,varX]
                
                plt.xlabel("Dimension / max(Dimension)")
                plt.ylabel('Efficiency [-], Eqn. (18)')



            print(individualOptimal)

            individualOptimalScaled= sc.transform(individualOptimal.reshape(1,-1))
            baselinedesignscaled= sc.transform(np.array(baselinedesign).reshape(1,-1))

            print("individual optimal:")  
            print(evaluate(individualOptimalScaled,gp))
            print("baseline:")
            print(evaluate(baselinedesignscaled,gp))

            plt.legend()


        isoplot=True
        if isoplot ==True:
            
            var1=1
            var2=2
            var3=0

 






        plt.show()



    if optimizeBool ==True:

        optimizeplot=False
        if optimizeplot:
            var1 =3
            var2 = 0

            Ndim = len(features)        
            N=100
            x_vec = np.linspace(-3, 3, N)
            y_vec = np.linspace(-3, 3, N)
            mat = np.meshgrid(x_vec, y_vec)
            numPoints = len(mat[0].flatten('C'))
            testPoints=np.ones((Ndim,numPoints))

            d1_plot= np.ones(numPoints)*85e+5 #Pm
            d2_plot= np.ones(numPoints)*45e+5 #Po
            d3_plot= np.ones(numPoints)*38e+5 #Ps
            d4_plot= np.ones(numPoints)*2.80e+5 #hm
            d5_plot= np.ones(numPoints)*4.3e+5 #hs
        
            vecunscaled = np.array([d1_plot,d2_plot,d3_plot,d4_plot,d5_plot])
            vecscaled=sc.transform(vecunscaled.T).T

            for j in range((Ndim)):
                if j == var1:
                    testPoints[j]=mat[0].flatten('C')
                elif j==var2:
                    testPoints[j]=mat[1].flatten('C')
                else:
                    testPoints[j] = vecscaled[j]

            pred_grid, std_grid =  gp.predict(testPoints.T, return_std=True)
            pred_grid= np.reshape(pred_grid, (N,N),order='C')
            std_grid = np.reshape(std_grid, (N,N),order='C')

            scaled=np.array([x_vec,x_vec,x_vec,x_vec,x_vec])
            unscaled=sc.inverse_transform(scaled.T)


            #------z axis --------
            cmin = -0.1
            cmax= 0.6
            ncolors=31
            levels =np.linspace(cmin,cmax,ncolors)

            diagram="ph"
            if diagram =="ph":
                
                pressurebar = np.divide(unscaled[:,var2],10**5)
                enthalpyKjkg = np.divide(unscaled[:,var1],1000)
                plt.contourf(enthalpyKjkg,pressurebar,pred_grid[:,:],levels=levels)

                cbar=plt.colorbar(label='Efficiency [-], eqn. (1)')
                plt.clim(cmin,cmax)
                
                plt.xlim([2.72e+2,3.5e+2])
                plt.xticks(np.arange(2.72e+2,3.5e+2, step=25))

                plt.ylim([60,150])
                plt.yticks(np.arange(60,150, step=20))
                
                plt.xlabel("Enthalpy [kJ/kg]")
                plt.ylabel("Pressure [Bar]")




        delta=1e-5
        imax = 20000

        Nstarts= 10
        starts = np.random.uniform(low=-1,high=1,size=(Nstarts,5))

        # startvalues_unscaled= np.array([0,45*10**5,38*10**5,0,430*10**3])
        # startvalues_scaled= sc.transform(startvalues_unscaled.reshape(1,-1)).T

        

        for i in range(Nstarts):
            # print("\nOptimization iteration: %d"%i)

            # starts[i,1] = startvalues_scaled[1]
            # starts[i,2] = startvalues_scaled[2]
            # starts[i,4] = startvalues_scaled[4]

            x_0 = starts[i,:]
            # print("Starting position: [%f, %f, %f, %f, %f]" %(x_0[0],x_0[1], x_0[2], x_0[3], x_0[4],  )   )

            limitL = [-2 ,-2, -2, -2, -2]
            limitH = [2 ,2, 2, 2, 2]

            limitL_unscaled= np.array([0,0,0,0,0])
            limitL= sc.transform(limitL_unscaled.reshape(1,-1)).T

            # limitH_unscaled= np.array([10**20,45.1*10**5,38.1*10**5,10**20,431*10**3])
            # limitH= sc.transform(limitH_unscaled.reshape(1,-1)).T


            
            
            x_new, eval_list, x_list= gradientDecent(x_0,delta,gp,imax,limitL,limitH)
            # print("Efficiency first %f, Efficiency optimized %f" %(eval_list[0],eval_list[-1]))
            # print("Optimized position: [%f, %f, %f, %f, %f]" %(x_new[0],x_new[1], x_new[2], x_new[3], x_new[4],  )   )

            optUnscaled = sc.inverse_transform(x_new)
            # print(optUnscaled)
            # print("Number of iterations: %d "%len(eval_list))

            print("%d %.4f %.4f %.3f %.1f %.3f %.2f %d" %(i,optUnscaled[0],optUnscaled[1],optUnscaled[2],optUnscaled[3],optUnscaled[4],eval_list[-1],len(eval_list)))

            x_list = np.array(x_list)
            x_list_unscaled=sc.inverse_transform(x_list)
            eval_list =np.array(eval_list)

            # print(x_list[:,0])
            
            # print(x_1[:][0])

            # plt.plot(x_list[:,0],eval_list.reshape(-1,))
            # plt.plot(x_list_unscaled[:,0],eval_list.reshape(-1,))
            # plt.plot(x_list_unscaled[:,3],x_list_unscaled[:,0])
            plt.plot(np.multiply(x_list_unscaled[:,1],10**3),np.multiply(x_list_unscaled[:,2],10**3),'k')
            










        plt.show()
            


    return kern, err

def efficiency(Pm, Po, Ps, hm, hs, ER, fluid):
    import CoolProp.CoolProp as CP

    ss = CP.PropsSI('S','P',Ps,'H',hs,fluid)
    sm = CP.PropsSI('S','P',Pm,'H',hm,fluid)
    hm_iso = CP.PropsSI('H','P',Po,'S',sm,fluid)
    hs_iso = CP.PropsSI('H','P',Po,'S',ss,fluid)

    eff = ER * (hs_iso - hs)/(hm - hm_iso)
    
    return eff

def evaluate(x,gp):
    return gp.predict(np.array(x).reshape(1,-1))
    
def grad(x,delta,gp):
    d = np.zeros(len(x))

    for i in range(len(x)):
        right = x.copy()
        left = x.copy()

        right[i] = x[i] + delta/2
        left[i] = x[i] - delta/2

        d[i] = (evaluate(right,gp) - evaluate(left,gp))/delta


    return d

def gradientDecent(x_0,delta,gp,imax,limitL,limitH):
    step =0.5
    tol = 1e-5

    # first iteration
    i=0
    x_prev=x_0
    gradient=np.array(grad(x_prev,delta,gp))
    diff = step*gradient

    for j in range(len(x_0)):
        if (x_prev[j] + diff[j]) > limitH[j]:

            diff[j]=0
        if x_prev[j] + diff[j] < limitL[j]:

            diff[j]=0

    x_new = x_prev + diff

    x_list=[x_new]
    eval_list=[evaluate(x_new,gp)]

    while i < imax and np.any(np.absolute(diff) >= tol):
        
        x_prev = x_new.copy()
        gradient=np.array(grad(x_prev,delta,gp))
        diff = step*gradient

        for j in range(len(x_0)):
            if (x_prev[j] + diff[j]) > limitH[j]:
                # print("limit")
                diff[j]=0
            if x_prev[j] + diff[j] < limitL[j]:
                # print("limit")
                diff[j]=0

        
        x_new = x_prev + diff
        # print(x_new)
        i=i+1

        x_list.append(x_new)
        eval_list.append(evaluate(x_new,gp))

    
    return x_new, eval_list, x_list



trainBool=True 
displayBool=True
optimizeBool=False
# csvname='Database_200pnt_reducedrange_proper.csv'
csvname='Database_design_200_withentropy.csv'
df = pd.read_csv(csvname)
# features = ["Pm", "Po", "Ps", "hm", "hs"]
features = ["DmotiveOut", "Dmix", "Lmix", "alphadiff", "DdiffOut"]

output = ["eff"]
seed = 2 #33

kernel, err = GPR_function(df,features,output,0.15,seed,displayBool,optimizeBool,trainBool)

print(kernel)

