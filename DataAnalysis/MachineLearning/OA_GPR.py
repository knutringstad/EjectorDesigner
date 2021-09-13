import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from joblib import load





dataset = "Geometry"
# dataset = "Operating conditions"

#### PREDICT A DESIGN PERFORMANCE
dmo = 0.002 
dmix = 0.005
lmix = 0.028
alpha_d = 3
dout = 0.03

#### PREDICT A OPERATING CONDITION PERFORMANCE
Pm = 90e+5
Po = 45e+5
Ps = 42e+5
hm = 2.9e+5
hs = 4.3e+5





if dataset== "Geometry":
    testpoint= [dmo, dmix ,lmix, alpha_d, dout]
elif dataset== "Operating conditions":
    testpoint= [Pm, Po ,Ps, hm, hs]

print("Loading pretrained gaussian process")
if dataset== "Geometry":
    gp= load('gp_geom.joblib') 
    sc = load('sc_geom.joblib')
    x_test = load('x_test_geom.joblib')
    x_train = load('x_train_geom.joblib')
    y_test = load('y_test_geom.joblib')
    y_train = load('y_train_geom.joblib')
elif dataset== "Operating conditions":
    gp= load('gp_oper.joblib') 
    sc = load('sc_oper.joblib')
    x_test = load('x_test_oper.joblib')
    x_train = load('x_train_oper.joblib')
    y_test = load('y_test_oper.joblib')
    y_train = load('y_train_oper.joblib')

kern = gp.kernel_
pred = gp.predict(x_test)

err=mean_absolute_error(y_test,pred)
err_square=mean_squared_error(y_test,pred)
print("Training data average absolute error: %.9f "%err)
print("Training data absolute square error: %.9f "%err_square)

vecunscaled = np.array(testpoint).reshape(-1,1)
vecscaled=sc.transform(vecunscaled.T)

pred, std = gp.predict(vecscaled,return_std=True)

print("Predicted efficiency: %f" %pred)
print("Predicted uncertainty: %f" %std)
