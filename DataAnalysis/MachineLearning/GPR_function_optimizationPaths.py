from math import fabs
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import csv
import statistics
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 
import CoolProp.CoolProp as CP
import matplotlib.tri as tri
from joblib import dump, load
from matplotlib import cm

def GPR_function(df,features, output, sample_size, seed, displayBool,optimizeBool,trainBool):

    # Drop crashed simulations, clean up
    df = df.dropna()
    df = df.drop(df.loc[df["CrashIndicator"]==1].index)
    df = df.drop(df.loc[df["uni_vel"]==0].index)
    df = df.drop(df.loc[abs(df["mfr_err"])>0.0001].index)

    df_ejector = df.copy() 
    df_ejector["Plift"] = df_ejector["Po"]- df_ejector["Ps"]
    df_ejector["ER"] = df_ejector["mfr_s"] / df_ejector["mfr_m"]
    # df_ejector = df_ejector.drop(df.loc[df_ejector["mfr_s"]<0.0].index)
    effv = df_ejector["ER"].copy()


    print(len(df_ejector))
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
        kernel = 0.2 * RBF(length_scale=lengthScale, length_scale_bounds=(1e-3, 3e1))  + WhiteKernel(noise_level=1e-2)#* RBF(length_scale=lengthScale, length_scale_bounds=(2e5, 5e6)) #  +0.1 * RBF(length_scale=1E+3, length_scale_bounds=(1e3, 1e4))
        # kernel = 0.2 * RBF(length_scale=lengthScale, length_scale_bounds=(1e-3, 3e1))  + WhiteKernel(noise_level=1e-2)* RBF(length_scale=lengthScale, length_scale_bounds=(1e-3, 3e1))
        gp = GaussianProcessRegressor(kernel=kernel,alpha=std_estimate**2,n_restarts_optimizer=20).fit(x_train, y_train)  #run fitting
        gp_dump = dump(gp,'gp.joblib')
        sc_dump = dump(sc,'sc.joblib')
        x_test_dump = dump(x_test,'x_test.joblib')
        x_train_dump = dump(x_train,'x_train.joblib')
        y_test_dump = dump(y_test,'y_test.joblib')
        y_train_dump = dump(y_train,'y_train.joblib')
        print("Model trained")
    else:
        print("Loading pretrained gaussian process")
        gp= load('gp.joblib') 
        sc = load('sc.joblib')
        x_test = load('x_test.joblib')
        x_train = load('x_train.joblib')
        y_test = load('y_test.joblib')
        y_train = load('y_train.joblib')

    kern = gp.kernel_
    pred = gp.predict(x_test)
    err=mean_absolute_error(y_test,pred)
    err_square=mean_squared_error(y_test,pred)
    print("Avg abs error: %.9f "%err)
    print("Avg square error: %.9f "%err_square)


    import time

    start = time.time()
    vecscaled=sc.transform(np.array([1,2,3,4,5]).reshape(-1,1).T).T
    print(evaluate([0,0,0,0,0],gp))
    end = time.time()
    print('time for a function call evaluation:')
    print(end - start)

    if displayBool:
        scatterplot=True
        if scatterplot==True:
            #------------------SCATTER PLOT FIGURE -------------------------
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(x_test[:,0],x_test[:,3], pred, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
            ax.scatter(x_test[:,0],x_test[:,3], y_test, c='g', s=50, zorder=10, edgecolors=(0, 0, 0))

            ax.scatter(x_train[:,0],x_train[:,3], y_train, c='b', s=50, zorder=10, edgecolors=(0, 0, 0))
            ax.set_zlim([0,0.5])

            # ax.set_xlabel("Mixing chamber diameter [-]")
            # ax.set_ylabel("Diffuser outlet diameter [-]")
            # ax.set_zlabel("mass conservation error [kg/s]")

        heatmapplot=True
        if heatmapplot==True:
            #------------------HEAT MAP FIGURE -------------------------
            var1 =3
            var2 = 0

            fig = plt.figure()

            Ndim = len(features)        

            N=100
            x_vec = np.linspace(-3, 3, N)
            y_vec = np.linspace(-3, 3, N)
            mat = np.meshgrid(x_vec, y_vec)
            numPoints = len(mat[0].flatten('C'))
            testPoints=np.ones((Ndim,numPoints))


            d1_plot= np.ones(numPoints)*85e+5 #Pm
            d2_plot= np.ones(numPoints)*52e+5 #Po
            d3_plot= np.ones(numPoints)*42e+5 #Ps
            d4_plot= np.ones(numPoints)*2.80e+5 #hm
            d5_plot= np.ones(numPoints)*4.3e+5 #hs

            # d1_plot= np.ones(numPoints)*0.002 #dmo / Lmch
            # d2_plot= np.ones(numPoints)*0.00257 #Dmix
            # d3_plot= np.ones(numPoints)*0.0234 #Lmix
            # d4_plot= np.ones(numPoints)*3.2#alphadiff
            # d5_plot= np.ones(numPoints)*0.04 #DdiffOut

            
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
            cmin = 0
            cmax= 0.2
            ncolors=11
            levels =np.linspace(cmin,cmax,ncolors)




            diagram="ph"
            if diagram =="ph":
                
                pressurebar = np.divide(unscaled[:,var2],10**5)
                enthalpyKjkg = np.divide(unscaled[:,var1],1000)
                plt.contourf(enthalpyKjkg,pressurebar,pred_grid[:,:],levels=levels,extend='both')

                cbar=plt.colorbar(label='Ejector efficiency [-], Eqn. (20)')
                plt.clim(cmin,cmax)
                
                p_vec = np.arange(50e5, CP.PropsSI('PCRIT','CO2'), 1000)
                CO2_hsat_v_vec = np.zeros(len(p_vec))
                CO2_hsat_l_vec = np.zeros(len(p_vec))

                for i,p in enumerate(p_vec):
                    CO2_hsat_l_vec[i] = CP.PropsSI('H','P',p_vec[i],'Q',0,'CO2')/1000
                    CO2_hsat_v_vec[i] = CP.PropsSI('H','P',p_vec[i],'Q',1,'CO2')/1000
                
                p_vec=np.divide(p_vec,10**5)


                satlinecolor="orange"
                plt.plot(CO2_hsat_l_vec,p_vec,satlinecolor)
                plt.plot(CO2_hsat_v_vec,p_vec,satlinecolor)
                plt.plot([CO2_hsat_l_vec[-1],CO2_hsat_v_vec[-1]],[p_vec[-1],p_vec[-1]],satlinecolor)

                plt.xlim([2.72e+2,3.5e+2])
                plt.xticks(np.arange(2.75e+2,3.6e+2, step=25))

                plt.ylim([60,150])
                plt.yticks(np.arange(60,150, step=20))
                
                plt.xlabel("Specific enthalpy [kJ/kg]")
                plt.ylabel("Pressure [bar]")

            elif diagram=="design":
                LabelList = ["Dmo [m]","Dmix [m]","Lmix [m]","AlphaD [deg]","Ddiff [m]"]

                plt.xlabel(LabelList[var1])
                plt.ylabel(LabelList[var2])


        stdplot=False
        if stdplot==True:
            # STANDARD DEVIATION PLOT

            fig = plt.figure()

            cmin = 0
            cmax= 0.1
            levels =np.linspace(cmin,cmax,ncolors)

            plt.contourf(unscaled[:,var1],unscaled[:,var2],std_grid[:,:],levels=levels,extend='both')
            plt.xlim([min(unscaled[:,var1]),max(unscaled[:,var1])])
            plt.ylim( [min(unscaled[:,var2]),max(unscaled[:,var2])])


            diagram="ph"
            if diagram =="ph":
                
                p_vec = np.arange(50e5, CP.PropsSI('PCRIT','CO2'), 1000)
                CO2_hsat_v_vec = np.zeros(len(p_vec))
                CO2_hsat_l_vec = np.zeros(len(p_vec))

                for i,p in enumerate(p_vec):
                    CO2_hsat_l_vec[i] = CP.PropsSI('H','P',p_vec[i],'Q',0,'CO2')
                    CO2_hsat_v_vec[i] = CP.PropsSI('H','P',p_vec[i],'Q',1,'CO2')
                
                
                plt.plot(CO2_hsat_l_vec,p_vec,'o')
                plt.plot(CO2_hsat_v_vec,p_vec,'o')
                plt.plot([CO2_hsat_l_vec[-1],CO2_hsat_v_vec[-1]],[p_vec[-1],p_vec[-1]],'o')
                plt.xlim([2.26e+5,3.7e+5])
                plt.xticks(np.arange(2.26e+5,3.7e+5, step=5e+4))
                
                plt.xlabel("Specific Enthalpy [j/kg]")
                plt.ylabel("Pressure [Pa]")

            elif diagram=="design":
                LabelList = ["Dmo [m]","Dmix [m]","Lmix [m]","AlphaD [deg]","Ddiff [m]"]

                plt.xlabel(LabelList[var1])
                plt.ylabel(LabelList[var2])

            plt.colorbar()
            plt.clim(cmin,cmax)


        individualOptPlot=False
        if individualOptPlot:
            # IDIVIDUAL OPTIMUM




            fig,ax= plt.subplots()
            individualOptimal = np.ones(5)

            legend=["Dmo","Dmix","Lmix","AlphaD","Ddiff"]
            colorlines=["red","green", "blue", "black", "purple"]

            for varX in range(5):
                N=50
                x_vec = np.linspace(-2, 2, N)
                
                numPoints = len(x_vec)
                testPoints=np.ones((Ndim,numPoints))

                d1_plot= np.ones(numPoints)*0.002 #dmo
                d2_plot= np.ones(numPoints)*0.02 #Dmix
                d3_plot= np.ones(numPoints)*0.26 #Lmix
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
                
                # ax.plot(dimlessX,pred_line+std_line[0],'-.',c=colorlines[varX],label='_Hidden label')
                # ax.plot(dimlessX,pred_line-std_line[0],'-.',c=colorlines[varX],label='_Hidden label')
                
                ax.set_ylim([-1,2])

                indexMax= np.argmax(pred_line)

                individualOptimal[varX]= unscaled[indexMax,varX]
                



            print(individualOptimal)
            individualOptimal =np.array( [1.77e-03, 3.82e-03, 1.347e-02,1.58e+01,6.51e-02] ) 

            individualOptimalScaled= sc.transform(individualOptimal.reshape(1,-1))
                
            print(evaluate(individualOptimalScaled,gp))

            plt.legend()

        plt.show()



    if optimizeBool ==True:

        optimizeplot=True
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
            cmin = 0
            cmax= 0.4
            ncolors=9
            levels =np.linspace(cmin,cmax,ncolors)

            diagram="ph"
            if diagram =="ph":
                
                pressurebar = np.divide(unscaled[:,var2],10**5)
                enthalpyKjkg = np.divide(unscaled[:,var1],1000)
                plt.contourf(enthalpyKjkg,pressurebar,pred_grid[:,:],levels=levels,extend='both')

                cbar=plt.colorbar(label='Ejector efficiency [-], Eqn. (20)')
                plt.clim(cmin,cmax)
                
                plt.xlim([2.7e+2,3.61e+2])
                plt.xticks(np.arange(2.75e+2,3.6e+2, step=25))

                plt.ylim([60,150])
                plt.yticks(np.arange(60,150, step=20))
                
                plt.xlabel("Specific enthalpy [kJ/kg]")
                plt.ylabel("Pressure [bar]")




        delta=1e-5
        imax = 20000

        Nstarts= 10
        starts = np.random.uniform(low=-2,high=2,size=(Nstarts,5))

        startvalues_unscaled= np.array([0,45*10**5,38*10**5,0,430*10**3])
        startvalues_scaled= sc.transform(startvalues_unscaled.reshape(1,-1)).T

        

        for i in range(Nstarts):
            # print("\nOptimization iteration: %d"%i)
            starts[i,1] = startvalues_scaled[1]
            starts[i,2] = startvalues_scaled[2]
            starts[i,4] = startvalues_scaled[4]
            x_0 = starts[i,:]
            # print("Starting position: [%f, %f, %f, %f, %f]" %(x_0[0],x_0[1], x_0[2], x_0[3], x_0[4],  )   )

            limitL = [-2 ,-2, -2, -2, -2]
            limitH = [2 ,2, 2, 2, 2]

            limitL_unscaled= np.array([10**-20,44.9*10**5,37.9*10**5,10**-20,429*10**3])
            limitL= sc.transform(limitL_unscaled.reshape(1,-1)).T

            limitH_unscaled= np.array([10**20,45.1*10**5,38.1*10**5,10**20,431*10**3])
            limitH= sc.transform(limitH_unscaled.reshape(1,-1)).T


            
            
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
            # plt.plot(np.divide(x_list_unscaled[:,3],10**3),np.divide(x_list_unscaled[:,0],10**5),'k')
            



            NscatterpointsSteps=10
            plt.plot(   np.divide(x_list_unscaled[:,3],10**3),np.divide(x_list_unscaled[:,0],10**5) ,'k', zorder=10+i)
            color_variable = np.divide(np.arange(len(x_list_unscaled[0:-1:NscatterpointsSteps,3])),len(x_list_unscaled[0:-1:NscatterpointsSteps,3]))
            plt.scatter(   np.divide(x_list_unscaled[0:-1:NscatterpointsSteps,3],10**3),np.divide(x_list_unscaled[0:-1:NscatterpointsSteps,0],10**5), s=20,    color=cm.plasma(color_variable)  , zorder=100+i  )
       






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
displayBool=False
optimizeBool=True
csvname='Database_Performance_600pnt.csv'
df = pd.read_csv(csvname)
features = ["Pm", "Po", "Ps", "hm", "hs"]
# features = ["DmotiveOut", "Dmix", "Lmix", "alphadiff", "DdiffOut"]

output = ["eff"]
seed =  40

kernel, err = GPR_function(df,features,output,0.15,seed,displayBool,optimizeBool,trainBool)

print(kernel)

# 600
# Avg abs error: 0.062966707 
# Avg square error: 0.022986046


# Avg abs error: 0.030355813 
# Avg square error: 0.003592155 

# Avg abs error: 0.124246449 
# Avg square error: 0.016124685 




# Avg abs error: 0.075855765
# Avg square error: 0.010074733
# 0.135**2 * RBF(length_scale=[0.995, 0.915, 0.744, 30, 30]) + WhiteKernel(noise_level=0.000195) * RBF(length_scale=[28.1, 3.09, 4.09, 8.9, 0.00248])

# Avg abs error: 0.040403203
# Avg square error: 0.003035981
# 0.101**2 * RBF(length_scale=[1, 0.485, 0.846, 3.35, 3.7]) + WhiteKernel(noise_level=0.00736) * RBF(length_scale=[12.4, 1.51, 0.00118, 0.617, 9.73])

# Avg abs error: 0.052683325 
# Avg square error: 0.005086014
# 0.324**2 * RBF(length_scale=[2.38, 3.36, 3.99, 3.54, 6.64]) + WhiteKernel(noise_level=0.00943) * RBF(length_scale=[0.00868, 0.00466, 0.599, 0.0457, 0.00903])