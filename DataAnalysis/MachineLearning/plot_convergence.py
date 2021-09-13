import matplotlib.pyplot as plt
import numpy as np

error_type="Abs"

if error_type=="MSE":
    ERavg_100 =0.171678
    effavg_100 =0.015762 
    mfrmavg_100=0.000007 
    mfrsavg_100 = 0.000421 
    unialphaavg_100=0.000002
    univelavg_100=0.000484 
    ds1avg_100 = 0.001991

    ERavg_300 =0.021667 
    effavg_300 =0.010960 
    mfrmavg_300=0.000000478
    mfrsavg_300 =0.000198 
    unialphaavg_300= 0.000000266 
    univelavg_300=0.000267 
    ds1avg_300 = 0.004023 

    ERavg_600 =0.012288 
    effavg_600 =0.002805 
    mfrmavg_600=0.000000418 
    mfrsavg_600 = 0.000059 
    unialphaavg_600=.000000141 
    univelavg_600=0.000244 
    ds1avg_600 = 0.003303 

elif error_type=="Abs":

    ERavg_100 = 0.292286 
    effavg_100 =0.105445 
    mfrmavg_100=0.001502 
    mfrsavg_100 = 0.014840
    unialphaavg_100=0.001099 
    univelavg_100=0.0005765
    ds1avg_100 = 0.032297

    ERavg_300 = 0.102076 
    effavg_300 =0.078322 
    mfrmavg_300=  0.010422  
    mfrsavg_300 = 0.002258
    unialphaavg_300= 0.000420 
    univelavg_300=0.016943 
    ds1avg_300 = 0.039742 

    ERavg_600 = 0.076658 
    effavg_600 =0.039257 
    mfrmavg_600=0.000512 
    mfrsavg_600 = 0.006038 
    unialphaavg_600= 0.000307
    univelavg_600=0.012124 
    ds1avg_600 = 0.039575 



# plt.figure()

ER = np.array([ERavg_100,ERavg_300,ERavg_600])
eff = np.array([effavg_100,effavg_300,effavg_600])
mfrm = np.array([mfrmavg_100,mfrmavg_300,mfrmavg_600])
mfrs = np.array([mfrsavg_100,mfrsavg_300,mfrsavg_600])
unialph = np.array([unialphaavg_100,unialphaavg_300,unialphaavg_600])
univel = np.array([univelavg_100,univelavg_300,univelavg_600])
ds = np.array([ds1avg_100,ds1avg_300,ds1avg_600])

ER=np.divide(ER,ERavg_100)
eff=np.divide(eff,effavg_100)
mfrm=np.divide(mfrm,mfrmavg_100)
mfrs=np.divide(mfrs,mfrsavg_100)
unialph=np.divide(unialph,unialphaavg_100)
univel=np.divide(univel,univelavg_100)
ds=np.divide(ds,ds1avg_100)


# plt.plot([100, 300, 600],ER)
# plt.plot([100, 300, 600],eff)
# plt.plot([100, 300, 600],mfrm)
# plt.plot([100, 300, 600],mfrs)
# # plt.plot([100, 300, 600],unialph)
# plt.plot([100, 300, 600],univel)
# plt.plot([100, 300, 600],ds)
# plt.ylim([0,4])
# plt.legend(["ER", "Efficiency", "MFR_m", "MFR_s", "Uniformity velocity", "Entropy growth mix-diff"],loc="best")
# plt.show()


N = 7
vals100 = [ERavg_100,effavg_100,mfrmavg_100,mfrsavg_100,univelavg_100,unialphaavg_100,ds1avg_100]
vals100nondim=np.divide(vals100,vals100)

vals300 = [ERavg_300,effavg_300,mfrmavg_300,mfrsavg_300,univelavg_300,unialphaavg_300,ds1avg_300]
vals300nondim=np.divide(vals300,vals100)

vals600 = [ERavg_600,effavg_600,mfrmavg_600,mfrsavg_600,univelavg_600,unialphaavg_600,ds1avg_600]
vals600nondim =np.divide(vals600,vals100)

ind = np.arange(N)    # the x locations for the groups
width = 0.31       # the width of the bars: can also be len(x) sequence


fig, ax = plt.subplots()


p1 = ax.bar(ind, vals100nondim, width, label='values_100')
p2 = ax.bar(ind+width, vals300nondim, width, label='values_300')
p3 = ax.bar(ind+width*2, vals600nondim, width, label='values_600')

ax.set_ylabel('MSE / MSE$_{100}$')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('ER', 'Efficiency', 'MFR$_m$','MFR$_s$','$\gamma_u$','$\gamma_{vel}$','$\Delta$Entropy') )

ax.legend( (p1[0], p2[0], p3[0]), ('100 points', '300 points', '600 points') )

plt.show()