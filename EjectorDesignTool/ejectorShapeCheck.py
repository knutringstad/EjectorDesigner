def EjectorShapeCheck(path_to_design_file):
    import csv
    import matplotlib.pyplot as plt
    import math
    import numpy as np

    shape = list(csv.reader(open(path_to_design_file)))

    Lmotive = float(shape[1][1])
    DmotiveIn = float(shape[2][1])
    DmotiveOut = float(shape[3][1])
    Dthroat = float(shape[4][1])
    alphaMotiveDiff = float(shape[5][1])
    alphaMotiveConv = float(shape[6][1])
    Lmch = float(shape[7][1])
    alphaSuction = float(shape[8][1])
    Dmix = float(shape[9][1])
    Lmix = float(shape[10][1])
    alphadiff = float(shape[11][1])
    DdiffOut = float(shape[12][1])
    Loutlet = float(shape[13][1])
    Dsuc = float(shape[14][1])
    ThicknessNozzle =float(shape[15][1])
    ThicknessNozzleOuter =float(shape[16][1])


    if ThicknessNozzleOuter <0:
        ThicknessNozzleOuter = ThicknessNozzle*2 
    
    Ldiff = (DdiffOut/2-Dmix/2)/(math.tan(math.radians(alphadiff/2)))
    Lthroat = Lmotive - (DmotiveOut/2-Dthroat/2)/(math.tan(math.radians(alphaMotiveDiff/2)))
    L1 = Lthroat - (-Dthroat/2+DmotiveIn/2)/(math.tan(math.radians(alphaMotiveConv/2)))

    if L1 <0 :
        print("in meshing")
        print(Lmotive)
        print(Lthroat)
        print(L1)
        raise Exception("error in geom setup: L1<0, increase L1")

    ys = np.ndarray(7,float)
    xs = np.ndarray(7,float)
    ym = np.ndarray(9,float)
    xm = np.ndarray(9,float)

    ym[1] = DmotiveIn/2
    ym[2] = DmotiveIn/2
    ym[3] = Dthroat/2
    ym[4] = DmotiveOut/2
    ym[8] = DmotiveIn/2+ThicknessNozzleOuter
    ym[7] = DmotiveIn/2+ThicknessNozzleOuter
    ym[6] = DmotiveIn/2+ThicknessNozzleOuter
    ym[5] = DmotiveOut/2+ThicknessNozzle

    xm [1] = 0
    xm [2] = L1
    xm [3] = Lthroat
    xm [4] = Lmotive
    xm [8] = 0

    xm [5] = Lmotive
    alphaMotiveOuter = alphaSuction # CHANGEABLE
    xm [6] = xm[5]-(ym[6]-ym[5])/math.tan(math.radians((alphaMotiveOuter/2)))
    xm [7] = xm[6]*0.9


    if Dsuc < 0:
        Dsuc = 2*(math.tan(math.radians(alphaSuction/2))*(Lmotive+Lmch-xs[2]) + Dmix/2)    
        xs[2]=xm [7]*0.9 
    else:
        Dsuc = 0.04
        xs[2]= Lmotive+Lmch-(Dsuc/2 -Dmix/2)/(math.tan(math.radians(alphaSuction/2)))
        if xs[2] <0 :
            raise Exception("error in geom setup: xs2<0, increase Lmotive size")
    
        
    ys[2] = Dsuc/2
    ys[1]= Dsuc/2
    xs[1]=0
    xs[3] = Lmotive+Lmch
    ys[3] = Dmix/2
    ys[4]=Dmix/2
    xs[4]= Lmotive+Lmch+Lmix
    ys[5]=DdiffOut/2
    xs[5]=Lmotive+Lmch+Lmix+Ldiff
    ys[6]=DdiffOut/2
    xs[6]=Lmotive+Lmch+Lmix+Ldiff+Loutlet



    plt.subplot(3,1,1)
    plt.plot(xs[1 :],ys[1 :],'black')
    plt.plot(xm[1 :],ym[1 :],'red')
    plt.axis('scaled')

    plt.subplot(3,1,2)
    plt.plot(xm[1 :],ym[1 :],'red')
    plt.axis('scaled')

    plt.subplot(3,1,3)
    plt.plot(xs[1 :],ys[1 :],'black')
    plt.axis('scaled')


    
    plt.show()



