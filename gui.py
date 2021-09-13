# img_viewer.py

import PySimpleGUI as sg
import os.path
import CoolProp.CoolProp as CP
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import matplotlib.pyplot as plt
import math


Pm = 11e+6
Po =  4.7e+6
Ps = 3.5e+6
Tm =   35
hm = CP.PropsSI('H','P',Pm,'T',273.15+Tm,'CO2')
Ts =  3
hs = CP.PropsSI('H','P',Ps,'T',273.15+Ts,'CO2')
Lmotive =0.220
DmotiveIn = 0.05 
DmotiveOut = 0.00781
Dthroat = 0.00572
alphaMotiveDiff = 4.0
alphaMotiveConv = 30.0
Lmch = 0.0095
alphaSuction = 40
Dmix = 0.0125
Lmix = 0.13
alphadiff = 5
DdiffOut = 0.05
Loutlet = 0.01
Dsuc= -1 #-1 indicates calculate based on internal params
ThicknessNozzle = 0.0005
ThicknessNozzleOuter= 0.005
p_vec = np.arange(6e5, CP.PropsSI('PCRIT','CO2'), 10000)
CO2_hsat_v_vec = np.zeros(len(p_vec))
CO2_hsat_l_vec = np.zeros(len(p_vec))

for i,p in enumerate(p_vec):
    CO2_hsat_l_vec[i] = CP.PropsSI('H','P',p_vec[i],'Q',0,'CO2')
    CO2_hsat_v_vec[i] = CP.PropsSI('H','P',p_vec[i],'Q',1,'CO2')
        

def runGui():

    case_list_column = [addVar('Pm','bar'), addVar('Ps','bar'), addVar('Po','bar'), addVar('Tm','deg C'), addVar('Ts','deg C'), 
    addVar('Dthroat','mm'), addVar('Dmix','mm'), addVar('Lmotive','mm'), addVar('DmotiveIn','mm'), addVar('DmotiveOut','mm'), 
    addVar('Lmch','mm'), addVar('Lmix','mm'),addVar('DdiffOut','mm'),addVar('Loutlet','mm'),addVar('Dsuc','mm'),
    addVar('alphaMotiveDiff','deg'), addVar('alphaMotiveConv','deg'),addVar('alphaSuction','deg'),addVar('alphadiff','deg'),
    addVar('ThicknessNozzle','mm'),addVar('ThicknessNozzleOuter','mm'),
    [sg.Submit('Update', key='-SUBMIT-')],
    ]

    image_viewer_column = [
        [sg.Canvas(key="-CANVAS_PH-")],
        [sg.Submit('Generate case', key='-GENERATE-')],
    ]

    layout = [
        [
            sg.Column(case_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
        ]
    ]

    window = sg.Window("AutoEjector Designer", layout)
    figure_agg_t = None

    initialUpdate(window)

    # Run the Event Loop
    while True:
        event, values = window.read()
        print(event, values)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == "-SUBMIT-":
            checkUpdate('Pm',values,window)
            checkUpdate('Ps',values,window)
            checkUpdate('Po',values,window)
            checkUpdate('Tm',values,window)
            checkUpdate('Ts',values,window)
            checkUpdate('Dthroat',values,window)
            checkUpdate('Dmix',values,window)
            checkUpdate('Lmotive',values,window)
            checkUpdate('DmotiveIn',values,window)
            checkUpdate('DmotiveOut',values,window)
            checkUpdate('Dthroat',values,window)
            checkUpdate('alphaMotiveDiff',values,window)
            checkUpdate('alphaMotiveConv',values,window)
            checkUpdate('Lmch',values,window)
            checkUpdate('alphaSuction',values,window)
            checkUpdate('Dmix',values,window)
            checkUpdate('Lmix',values,window)
            checkUpdate('alphadiff',values,window)
            checkUpdate('DdiffOut',values,window)
            checkUpdate('Loutlet',values,window)
            checkUpdate('Dsuc',values,window)
            checkUpdate('ThicknessNozzle',values,window)
            checkUpdate('ThicknessNozzleOuter',values,window)

            if figure_agg_t:
                delete_figure_agg(figure_agg_t)

            fig_ph = make_ph_fig()

            figure_agg_t = draw_figure(window['-CANVAS_PH-'].TKCanvas, fig_ph)

        if event == "-GENERATE-":
            generate('GUI_test')

    window.close()



def generate(Name):
    from Cases.CaseSetup import MakeCase
    MakeCase(Name)
    global Tm, Pm, Ps, Po, Ts, hs, hm, Lmotive, DmotiveIn , DmotiveOut , Dthroat , alphaMotiveDiff, alphaMotiveConv, Lmch,alphaSuction, Dmix, Lmix, alphadiff, DdiffOut, Loutlet, Dsuc, ThicknessNozzle, ThicknessNozzleOuter

    f=open('./Cases/%s/Settings/BaselineConditions.py' %Name,'w')
    f.write('import CoolProp.CoolProp as CP\n')
    f.write('Pm = %f\n' %Pm)
    f.write('Po = %f\n' %Po)
    f.write('Ps = %f\n' %Ps)

    f.write('hm =   CP.PropsSI(\'H\',\'P\',Pm,\'T\',273.15+%f,\'CO2\')\n' %Tm)
    f.write('hs =  CP.PropsSI(\'H\',\'P\',Ps,\'T\',273.15+%f,\'CO2\')\n'%Ts)

    f.write('Boundary_conditions = [Pm, Po, Ps, hm, hs]\n')

    f.write('Lmotive =%f\n' %Lmotive)
    f.write('DmotiveIn = %f\n ' %DmotiveIn)
    f.write('DmotiveOut = %f \n' %DmotiveOut)
    f.write('Dthroat = %f\n' %Dthroat)
    f.write('alphaMotiveDiff = %f\n' %alphaMotiveDiff)
    f.write('alphaMotiveConv = %f\n' %alphaMotiveConv)
    f.write('Lmch = %f\n' %Lmch)
    f.write('alphaSuction = %f\n' %alphaSuction)
    f.write('Dmix = %f\n' %Dmix)
    f.write('Lmix = %f\n' %Lmix)
    f.write('alphadiff = %f\n'%alphadiff)
    f.write('DdiffOut = %f\n'%DdiffOut)
    f.write('Loutlet = %f\n'%Loutlet)
    f.write('Dsuc= %f #-1 indicates calculate based on internal params\n' %Dsuc)
    f.write('ThicknessNozzle = %f\n' %ThicknessNozzle)
    f.write('ThicknessNozzleOuter= %f\n' %ThicknessNozzleOuter)

    f.write('Geometry_parameters= [Lmotive, DmotiveIn , DmotiveOut , Dthroat , alphaMotiveDiff, alphaMotiveConv, Lmch,alphaSuction, Dmix, Lmix, alphadiff, DdiffOut, Loutlet, Dsuc, ThicknessNozzle, ThicknessNozzleOuter]\n')

    f.write('InputStandard = Boundary_conditions + Geometry_parameters \n')

    f.close()


def initialUpdate(window):
    window.read()
    window['-'+'Pm'+'-'].update(Pm/1e5)
    window['-'+'Po'+'-'].update(Po/1e5)
    window['-'+'Ps'+'-'].update(Ps/1e5)
    window['-'+'Tm'+'-'].update(Tm)
    window['-'+'Ts'+'-'].update(Ts)
    window['-'+'Lmotive'+'-'].update(Lmotive/1e-3)
    window['-'+'DmotiveIn'+'-'].update(DmotiveIn/1e-3)
    window['-'+'DmotiveOut'+'-'].update(DmotiveOut/1e-3)
    window['-'+'Dthroat'+'-'].update(Dthroat/1e-3)
    window['-'+'alphaMotiveDiff'+'-'].update(alphaMotiveDiff)
    window['-'+'alphaMotiveConv'+'-'].update(alphaMotiveConv)
    window['-'+'Lmch'+'-'].update(Lmch/1e-3)
    window['-'+'alphaSuction'+'-'].update(alphaSuction)
    window['-'+'Dmix'+'-'].update(Dmix/1e-3)
    window['-'+'Lmix'+'-'].update(Lmix/1e-3)
    window['-'+'alphadiff'+'-'].update(alphadiff)
    window['-'+'DdiffOut'+'-'].update(DdiffOut/1e-3)
    window['-'+'Loutlet'+'-'].update(Loutlet/1e-3)
    window['-'+'Dsuc'+'-'].update(Dsuc/1e-3)
    window['-'+'ThicknessNozzle'+'-'].update(ThicknessNozzle/1e-3)
    window['-'+'ThicknessNozzleOuter'+'-'].update(ThicknessNozzleOuter/1e-3)



def checkUpdate(var,values,window):
    if values['-'+var+'INPUT'+'-']:# and values['-'+var+'INPUT'+'-'] [:-1] in ('0123456789.-'):
        window['-'+var+'-'].update(values['-'+var+'INPUT'+'-'])
        global Tm, Pm, Ps, Po, Ts, hs, hm, Lmotive, DmotiveIn , DmotiveOut , Dthroat , alphaMotiveDiff, alphaMotiveConv, Lmch,alphaSuction, Dmix, Lmix, alphadiff, DdiffOut, Loutlet, Dsuc, ThicknessNozzle, ThicknessNozzleOuter
        if var == 'Pm':
            Pm = float(values['-'+var+'INPUT'+'-'])*10**5
        if var == 'Po':
            Po = float(values['-'+var+'INPUT'+'-'])*10**5
        if var == 'Ps':
            Ps = float(values['-'+var+'INPUT'+'-'])*10**5
        if var == 'Tm':
            Tm =float(values['-'+var+'INPUT'+'-'])
            hm= CP.PropsSI('H','P',Pm,'T',273.15+Tm,'CO2')
        if var == 'Ts':
            Ts =float(values['-'+var+'INPUT'+'-'])
            hs = CP.PropsSI('H','P',Ps,'T',273.15+Ts,'CO2')
        if var == 'Dthroat':
            Dthroat =float(values['-'+var+'INPUT'+'-'])*10**-3
        if var == 'Dmix':
            Dmix =float(values['-'+var+'INPUT'+'-'])*10**-3
        if var == 'Lmotive':
            Lmotive =float(values['-'+var+'INPUT'+'-'])*10**-3
        if var == 'DmotiveIn':
            DmotiveIn =float(values['-'+var+'INPUT'+'-'])*10**-3
        if var == 'DmotiveOut':
            DmotiveOut =float(values['-'+var+'INPUT'+'-'])*10**-3
        if var == 'Lmch':
            Lmch =float(values['-'+var+'INPUT'+'-'])*10**-3
        if var == 'Lmix':
            Lmix =float(values['-'+var+'INPUT'+'-'])*10**-3
        if var == 'DdiffOut':
            DdiffOut =float(values['-'+var+'INPUT'+'-'])*10**-3
        if var == 'Loutlet':
            Loutlet =float(values['-'+var+'INPUT'+'-'])*10**-3 
        if var == 'Dsuc':
            Dsuc =float(values['-'+var+'INPUT'+'-'])*10**-3
        if var == 'ThicknessNozzle':
            ThicknessNozzle =float(values['-'+var+'INPUT'+'-'])*10**-3
        if var == 'ThicknessNozzleOuter':
            ThicknessNozzleOuter =float(values['-'+var+'INPUT'+'-'])*10**-3
        if var == 'alphaMotiveDiff':
            alphaMotiveDiff =float(values['-'+var+'INPUT'+'-'])
        if var == 'alphaMotiveConv':
            alphaMotiveConv =float(values['-'+var+'INPUT'+'-'])
        if var == 'alphaSuction':
            alphaSuction =float(values['-'+var+'INPUT'+'-'])
        if var == 'alphadiff':
            alphadiff =float(values['-'+var+'INPUT'+'-'])

    else:
        print('invalid input for variable %s' %var)

def addVar(var,unit):
    return [
        sg.Text(var + ' : ', size=(15,1)),
        sg.Input(size=(8, 1),key='-'+var+'INPUT'+'-'),
        sg.Text(size=(8,1), key='-'+var+'-'),
        sg.Text("["+unit+"]"),
        ]

def make_ph_fig():
    global hm, hs, Pm, Ps
    import matplotlib.pyplot as plt

    ax1 = plt.subplot(211)
    ax1.scatter([hm,hs],[Pm,Ps])
    ax1.plot(CO2_hsat_l_vec,p_vec)
    ax1.plot(CO2_hsat_v_vec,p_vec)

    global Lmotive, DmotiveIn , DmotiveOut , Dthroat , alphaMotiveDiff, alphaMotiveConv, Lmch,alphaSuction, Dmix, Lmix, alphadiff, DdiffOut, Loutlet, Dsuc, ThicknessNozzle, ThicknessNozzleOuter

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
        xs[2]=xm [7]*0.9 
        Dsuc = 2*(math.tan(math.radians(alphaSuction/2))*(Lmotive+Lmch-xs[2]) + Dmix/2)    
        
    else:
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

    
    plt.subplot(212)

    plt.plot(xm[1:-1], ym[1:-1])
    plt.plot(xs[1:-1], ys[1:-1])


    fig_design = plt.gcf()
    
    return fig_design



def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')





runGui()