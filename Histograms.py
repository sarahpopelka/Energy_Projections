import csv
import numpy as np
import ipywidgets as widgets
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

def func(b=None):
    plt.clf()
    use=Use.value
    rawdat=picker(use)
    if Metric.value=='Mean Monthly Use (BTU)':
        dat=[i[0] for i in rawdat]
    elif Metric.value=='Mean Monthly EUI (BTU)':
        dat=[i[0]/i[1] for i in rawdat]
    plt.hist(dat,bins='fd')
    plt.title('Histogram of Energy Use by '+ str(use))
    if m.value:
        plt.axvline(np.mean(dat), color='red', linestyle='solid', linewidth=2,label='Mean: '+str(np.mean(dat)))
    if p.value:
        childs=[Use, Metric ,m,p,perc, butt]
        plt.axvline(np.percentile(dat,perc.value), color='red', linestyle='dashed', linewidth=2,label=str(perc.value)+'th Percentile: ' + str(np.percentile(dat,perc.value)))
    else:
        childs=[Use, Metric ,m,p, butt]
    plt.legend(loc='upper right')
    plt.xlabel(str(Metric.value))
    plt.show()
    
Use=widgets.Dropdown(options=Desclist,description='Use Type:')
Metric=widgets.Dropdown(options=['Mean Monthly Use (BTU)','Mean Monthly EUI (BTU)'],description='Metric:')

m=widgets.Checkbox(value=False,description='Show Mean')
p=widgets.Checkbox(value=False,description='Show Percentile')
perc=widgets.BoundedIntText(value=0,min=0,max=100,description='Percentile:')

butt=widgets.Button(description='Plot')

def plot_on_click(b):
    clear_output()
    if p.value:
        display(widgets.VBox(children=[Use, Metric ,m,p,perc, butt]))
    else:
        display(widgets.VBox(children=[Use, Metric ,m,p, butt]))
    func()
    
butt.on_click(plot_on_click)

display(widgets.VBox(children=[Use, Metric ,m,p, butt]))