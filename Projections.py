###Sarah Popelka
###City of Santa Monica, 2018
###Code to help with Energy Projections, based on sector EUI caps

import csv
import numpy as np
import ipywidgets as widgets
from sympy import var
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from IPython.display import clear_output

infile='Matched1to1New.csv'

def read_edata(filename, column_number):
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        mylist = []
        for row in csvreader:
            try:
                mylist.append(str(row[column_number]))
            except:
                mylist.append(str(row[column_number]))
    return mylist

def picker(etype):
    vallist=[]
    rows=len(Types)
    for i in range(0,rows):
        if eTable[0,i]==etype:
            vallist.append(eTable[:,i])
    histdat=[(3412.14*float(i[2]),float(i[1])) for i in vallist if float(i[1])>0 and float(i[2])>0]
    return histdat

def lineargrowth(init,growth,rates,years,negs=True):
    lglist=[init]
    for i in range(0,years):
        if negs:
            if lglist[i]+lglist[i]*rates+growth>0:
                lglist.append(lglist[i]+lglist[i]*rates+growth)
            else:
                lglist.append(0)
        else:
            lglist.append(lglist[i]+lglist[i]*rates+growth)
    return lglist

def lineardecline(init,growth,rates,years,negs=True):
    lglist=[init]
    for i in range(0,years):
        if negs:
            if lglist[i]+lglist[i]*rates-growth>0:
                lglist.append(lglist[i]+lglist[i]*rates-growth)
            else:
                lglist.append(0)
        else:
            lglist.append(lglist[i]+lglist[i]*rates-growth)
    return lglist

sumkwh=read_edata(infile,5)
category=read_edata(infile,1)
types=read_edata(infile,2)
sqft=read_edata(infile,3)
meankwh=read_edata(infile,4)
Sumkwh=[str(i) for i in sumkwh[1:]]
Category=[str(i) for i in category[1:]]
Types=[str(i) for i in types[1:]]
Sqft=[str(i) for i in sqft[1:]]
Meankwh=[str(i) for i in meankwh[1:]]

eTable=np.array([Types,Sqft,Meankwh,Sumkwh])

def getKey(item):
    return item[1]

Desclist1=sorted(list(set([(Category[i],Types[i]) for i in range(0,len(Category))])),key=getKey)
Desclist=['']
for d in Desclist1:
    if d[0]!='Residential':
        Desclist.append(d[1])

def nonzero(in1,in2):
    for i in range(0, len(in1)):
        if in1[i]==0:
            in2[i]=in2[i-1]+in1[i-1]
    return in1, in2

def func(b=None):
    plt.clf()
    sectorsy=[i.children[0] for i in tab.children]
    percsy=[i.children[1] for i in tab.children]
    crsy=[i.children[2] for i in tab.children]
    sqft_capsy=[i.children[3] for i in tab.children]
    
    sectors=[i.value for i in sectorsy]
    percs=[i.value for i in percsy]
    crs=[i.value for i in crsy]
    sqft_caps=[i.value for i in sqft_capsy]
    
    tstop=ts.value
    t=np.arange(1,tstop)
    
    residential_rate=-0.0057174
    commercial_rate=0.00001812
    naturalgas_rate=-0.0002974

    def E_prime(E,t, commercial_rate):
        dedt=E*commercial_rate
        return dedt
    def N_prime(N,t, naturalgas_rate):
        dndt=N*naturalgas_rate
        return dndt
    def R_prime(R,t,residential_rate):
        drdt=R*residential_rate
        return drdt

    og_E=2017970000000

    init_N=2303450000000
    init_R=754254000000
    
    ode_N=odeint(N_prime,init_N,t,args=(naturalgas_rate,))
    ode_R=odeint(R_prime, init_R, t,args=(residential_rate,))
    ode_baseline=odeint(E_prime,og_E,t,args=(commercial_rate,))
    
    above50=[]
    below50=[]
    badsecs=[]
    
    for i in range(0,len(sectors)):
        if sectors[i]!='':
            plt.clf()
            dat=picker(sectors[i])
            EUIs=[float(d[0]/d[1]) for d in dat]
            cutoff=np.percentile(EUIs,percs[i])

            if len(dat)>5:
                annual_above=[d for d in dat if float(d[0]/d[1])>cutoff and d[1]>=sqft_caps[i]]
                annual_below=[d for d in dat if float(d[0]/d[1])<=cutoff or d[1]<sqft_caps[i]]

                averageuse=np.mean([float(a[0]) for a in annual_above])
                averageeui=np.mean([float(i[0]/i[1]) for i in annual_above])
                averagesqft=np.mean([float(i[1]) for i in annual_above])

                complyrate=crs[i]/100.0   


                init_above50=sum([a[0] for a in annual_above])*12
                init_sub50=sum([a[0] for a in annual_below])*12

                Sub50_comply=complyrate*len(annual_above)

                ode_Sub50=lineargrowth(init_sub50,Sub50_comply*(cutoff*averagesqft*12),commercial_rate,tstop)
                ode_above50=lineardecline(init_above50,Sub50_comply*(averagesqft*averageeui*12),commercial_rate,tstop)

                ode_above50_nz,ode_Sub50_nz=nonzero(ode_above50,ode_Sub50)
                
                zcount=ode_above50_nz.count(0)
                new_ode_Sub50_nz=ode_Sub50_nz[0:len(ode_Sub50_nz)-zcount]
                
                future_Sub50=5

                above50.append(ode_above50_nz)
                below50.append(ode_Sub50_nz)

            else:
                badsecs.append(sectors[i])
        
    if len(badsecs)==0:
        init_E=og_E-(sum([i[0] for i in above50])+sum([i[0] for i in below50]))
        ode_E=odeint(E_prime,init_E,t,args=(commercial_rate,))
        
        if showall.value:
            baseline_projection=ode_baseline[-1]
            adjusted_projection=ode_E[-1]+sum([i[-1] for i in below50])+sum([i[-1] for i in above50])
            total_2015=og_E
        else:
            baseline_projection=ode_baseline[-1]+ode_N[-1]+ode_R[-1]
            adjusted_projection=ode_E[-1]+ode_N[-1]+ode_R[-1]+sum([i[-1] for i in below50])+sum([i[-1] for i in above50])
            total_2015=og_E+init_N+init_R

        realvals=(0, adjusted_projection)
        adjusted=(total_2015,baseline_projection-adjusted_projection)

        ind=np.arange(2)

        p1=plt.bar(ind,adjusted,1,bottom=realvals,color='#f09f44')
        p2=plt.bar(ind,realvals,1,color='#38bab0')

        newyear=2015+tstop

        plt.ylabel('BTU')
        plt.xticks(ind,('2015',str(newyear)))
        plt.show()
            
    else:
        secstr=''
        for i in range(0,len(badsecs)):
            secstr+=', '+badsecs[i]
        print('Insufficient data for' + secstr[1:])
            
c1=  widgets.VBox([widgets.Dropdown(options=Desclist,description='Use Type:'),widgets.BoundedFloatText(value=50.0,min=0.0,max=100.0,step=1, description='EUI Percentile'),widgets.BoundedFloatText(value=10,min=0,max=100, step=1, description='Comply Rate (%)'),widgets.IntSlider(value=10000,min=0,max=50000,step=5000,description='SqFt', disabled=False,continuous_update=False,orientation='horizontal',readout=True,readout_format='d')])
c2=  widgets.VBox([widgets.Dropdown(options=Desclist,description='Use Type:'),widgets.BoundedFloatText(value=50.0,min=0.0,max=100.0,step=1, description='EUI Percentile'),widgets.BoundedFloatText(value=10,min=0,max=100, step=1, description='Comply Rate (%)'),widgets.IntSlider(value=10000,min=0,max=50000,step=5000,description='SqFt', disabled=False,continuous_update=False,orientation='horizontal',readout=True,readout_format='d')])
c3=  widgets.VBox([widgets.Dropdown(options=Desclist,description='Use Type:'),widgets.BoundedFloatText(value=50.0,min=0.0,max=100.0,step=1, description='EUI Percentile'),widgets.BoundedFloatText(value=10,min=0,max=100, step=1, description='Comply Rate (%)'),widgets.IntSlider(value=10000,min=0,max=50000,step=5000,description='SqFt', disabled=False,continuous_update=False,orientation='horizontal',readout=True,readout_format='d')])
c4=  widgets.VBox([widgets.Dropdown(options=Desclist,description='Use Type:'),widgets.BoundedFloatText(value=50.0,min=0.0,max=100.0,step=1, description='EUI Percentile'),widgets.BoundedFloatText(value=10,min=0,max=100, step=1, description='Comply Rate (%)'),widgets.IntSlider(value=10000,min=0,max=50000,step=5000,description='SqFt', disabled=False,continuous_update=False,orientation='horizontal',readout=True,readout_format='d')])

children=[c1,c2,c3,c4]

tab=widgets.Tab()

tab.children=children  
for i in range(0,len(children)):
    tab.set_title(i, 'Type '+str(i+1))
    
butt=widgets.Button(description='Run')

ts=widgets.IntSlider(value=30,min=2,max=50,step=1,description='Years', disabled=False,continuous_update=False,orientation='horizontal',readout=True,readout_format='d')

showall=widgets.Checkbox(value=False,description='Commercial Electricity Only')

def plot_on_click(b):
    clear_output()
    display(widgets.VBox(children=[tab,ts,showall,butt]))
    func()

butt.on_click(plot_on_click)

display(widgets.VBox(children=[tab,ts,showall,butt]))
#bigbox=widgets.VBox([tab,widgets.IntSlider(value=30,min=2,max=50,step=1,description='Years', disabled=False,continuous_update=False,orientation='horizontal',readout=True,readout_format='d')])

#butt=widgets.interact_manual(func,varys=tab, tstop=widgets.IntSlider(value=30,min=2,max=50,step=1,description='Years', disabled=False,continuous_update=False,orientation='horizontal',readout=True,readout_format='d'))
