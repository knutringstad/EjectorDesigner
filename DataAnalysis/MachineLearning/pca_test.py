import numpy as np
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, same_color
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

output = "ER"


csvname='Database_design_200_withentropy.csv'
features = ["DmotiveOut", "Dmix", "Lmix", "alphadiff", "DdiffOut"]
df = pd.read_csv(csvname)

sample_size=0.15
seed=2

# Drop crashed simulations, clean up

df = df.dropna()
df = df.drop(df.loc[df["CrashIndicator"]==1].index)
df = df.drop(df.loc[df["uni_vel"]==0].index)
df = df.drop(df.loc[abs(df["mfr_err"])>0.0001].index)


if output[0].startswith("ds_"):
    df = df.drop(df.loc[df["mfr_s"]<0].index)

print(len(df))
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
    # effv[index]=efficiency(Pm, Po, Ps, hm ,hs, ER, "CO2")
    
    if effv[index]<0:
        effv[index]=0

# print(effv)
df_ejector["eff"] = effv


x = df_ejector[features]
y = df_ejector[output]


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=sample_size,random_state=seed)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


pca = PCA(n_components=3)
pca.fit(X_train)


print(pca.components_)
# print(pca.get_params())

print(pca.explained_variance_ratio_)