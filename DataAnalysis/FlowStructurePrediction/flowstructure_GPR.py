
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel ,RationalQuadratic
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

folder='.'
numfiles=3
k=0
N= 1200

P_all = []
X_all = []
Po_all= []

Average_over_N = 10

for i in range(3,6):
    P = np.zeros((N,1))
    X = np.zeros((N,1))
    Po= np.zeros((N,1))
    filename='U_x_ML3_%d' %(i)

    if os.path.isfile(filename): 
        datContent = [i.strip().split() for i in open(filename).readlines()]
        
        k=0
        for j,line in enumerate(datContent[4:-2]):
            
            P[k]=datContent[j+4][1]
            X[k]=datContent[j+4][0]
            Po[k] = 40e+5 +(i-3)*1e+5
        
            k=k+1
        
        P=P[:k-1]
        X=X[:k-1]
        Po=Po[:k-1]

        X= np.mean(np.array(X[:k-k%Average_over_N]).reshape(-1,Average_over_N),axis=1)
        P= np.mean(np.array(P[:k-k%Average_over_N]).reshape(-1,Average_over_N),axis=1)
        Po= np.mean(np.array(Po[:k-k%Average_over_N]).reshape(-1,Average_over_N),axis=1)

        P_all.append(P)
        X_all.append(X)
        Po_all.append(Po)



P_all = np.array(P_all).flatten()
X_all = np.array(X_all).flatten()
Po_all = np.array(Po_all).flatten()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_all,Po_all, P_all, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))


# Data= np.concatenate([X_all,Po_all],axis=0)
Data = np.array([X_all,Po_all,P_all]).T
print(np.shape(Data))

df =  pd.DataFrame(Data, columns=['x_pos','Po','P'])

print(df.head())
if True:

    #GPR

    sample_size =0.15
    seed = 2

    features=['x_pos','Po']
    output = 'P'

    x_train, x_test, y_train, y_test = train_test_split(df[features],df[output], test_size=sample_size,random_state=seed)



    x_train = np.array(x_train)
    x_test = np.array(x_test)


    # Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)


    lengthScale = np.ones(len(features))*0.5
    std_estimate= 0.000

    kernel = ConstantKernel(1e+6) * RBF(length_scale=lengthScale)  + WhiteKernel(noise_level=1e-2) 
    # kernel = ConstantKernel(1e+6) * RBF(length_scale=1) + RationalQuadratic(length_scale=1.0, alpha=1.5) + WhiteKernel(noise_level=1e-2) 

    gp = GaussianProcessRegressor(kernel=kernel,alpha=std_estimate**2,n_restarts_optimizer=2,normalize_y=True).fit(x_train, y_train)  #run fitting


    kern = gp.kernel_
    print(kern)
    pred = gp.predict(x_test)
    err=mean_absolute_error(y_test,pred)
    err_square=mean_squared_error(y_test,pred)

    print(err)


    Ndim = len(features)        

    N_points=40
    x_vec = np.linspace(-2, 2, N_points)
    y_vec = np.linspace(-2, 2, N_points)
    mat = np.meshgrid(x_vec, y_vec)
    X_,Y_ = np.meshgrid(x_vec, y_vec)

    numPoints = len(mat[0].flatten('C'))
    testPoints=np.ones((Ndim,numPoints))

    testPoints[0] = mat[0].flatten('C')
    testPoints[1] = mat[1].flatten('C')







    pred_grid, std_grid = gp.predict(testPoints.T, return_std=True)

    pred_grid= np.reshape(pred_grid, (N_points,N_points),order='C')

    # plt.contourf(x_vec,y_vec,pred_grid[:,:],extend='both')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_zscale('log')
    surf = ax.plot_surface(X_,Y_, pred_grid[:,:],alpha=0.3)

    scaled= sc.transform(np.array([X_all,Po_all]).T).T
    ax.scatter(scaled[0],scaled[1], P_all, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))


    # ax.set_zlim(35e+5, 45e+5)

    plt.show()




    # X_ = np.linspace(-0.03, 0.03, 100)
    # x_ = sc.transform(X_[:, np.newaxis])
    # plt.plot(X_, y_mean, 'k')
    # plt.fill_between(X_, y_mean - y_std, y_mean + y_std,alpha=0.2, color='k')
    # plt.plot(X,P)