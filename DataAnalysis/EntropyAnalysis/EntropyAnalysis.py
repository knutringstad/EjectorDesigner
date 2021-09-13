
import pandas as pd
import os
import numpy as np
import matplotlib 

# folder='./DataAnalysis/EntropyAnalysis/Data_ML_Design_2'

folder='Data_ML_Design_2'
Ndata = 200

entropy = np.ones((Ndata,10))

for i in range(Ndata):
    filename='%s/entropy_integrals_%d.dat' %(folder,i)
    if os.path.isfile(filename): 
        datContent = [i.strip().split() for i in open(filename).readlines()]

        entropy[i,0] = float(datContent[5][1])
        entropy[i,1] =float(datContent[6][1])
        entropy[i,2] = float(datContent[7][1])
        entropy[i,3] =float(datContent[8][1])
        entropy[i,4] =float(datContent[9][1])
        entropy[i,5] =float(datContent[10][1])
        entropy[i,6]=float(datContent[11][1])
        entropy[i,7] =float(datContent[12][1])
        entropy[i,8] =float(datContent[13][1])
        entropy[i,9] = float(datContent[14][1])

df = pd.DataFrame(entropy, columns=['s_out','s_m_inlet','s_s_in','s_diff_out','s_mch','s_mix_in','s_mix_out','s_m_in','s_m_t','s_m_o'])

DatabaseName="Database_200pnt_reducedrange_proper"
df_big= pd.read_csv('%s.csv' %(DatabaseName))

df_merge=pd.concat([df_big, df], axis=1)

df_merge["ds_diff"] = df_merge["s_diff_out"] - df_merge["s_mix_out"]
df_merge["ds_mix"] = df_merge["s_mix_out"] - df_merge["s_mix_in"]
df_merge["ds_mot"] = df_merge["s_m_o"] - df_merge["s_m_in"]



print(df_merge)

df_merge.to_csv("Database_design_200_withentropy.csv")