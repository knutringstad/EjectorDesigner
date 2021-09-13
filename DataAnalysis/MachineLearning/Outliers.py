import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import csv
import statistics
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 
import CoolProp.CoolProp as CP
from joblib import dump, load

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from matplotlib import cm


csvname='Database_200pnt_reducedrange_proper.csv'
df = pd.read_csv(csvname)


df = df.dropna()
df = df.drop(df.loc[df["CrashIndicator"]==1].index)

df["ER"] = df["mfr_s"] / df["mfr_m"]
# df = df.drop(df.loc[df["uni_vel"]==0].index)


features =  ["ER"]

df2=df[features]
scaler = MinMaxScaler() 
df2 = pd.DataFrame(scaler.fit_transform(df2),columns=features)


outlier_detection = DBSCAN( eps = .4,  metric="euclidean",  min_samples = 10, n_jobs = -1)


clusters = outlier_detection.fit_predict(df2)


features2 = ["DmotiveOut", "Dmix", "Lmix", "alphadiff", "DdiffOut"]

df3= df[clusters==-1]

print(df3[:][features2])


cmap = cm.get_cmap('Set1')
df.plot.scatter(x='Lmix',y='Dmix', c=clusters, cmap=cmap, colorbar = False)
plt.show()