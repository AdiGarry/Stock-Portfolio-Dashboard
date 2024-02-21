import yfinance as yf
import numpy as np
import pandas as pd
import csv
from datetime import date
import datetime as dt
import math
from bokeh.io import curdoc,show 
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import TextInput, Button, DatePicker, MultiChoice,Select



enddate = date.today() 
stock_name=[]
stock_avgprice=[]
stock_quantity=[]
date_bought = []
data=[]
data1=[]
data2=[]
data3=[]

holding= open("C:/Users/adity/OneDrive/Desktop/Stock Portfolio Python\holding.txt")
csv_f = csv.reader(holding)


for row in csv_f:
    stock=(row[0])
    stock_name.append(stock)
    date=(row[1])
    date_bought.append(date)
    price=float((row[2]))
    stock_avgprice.append(price)
    no=int(row[3])
    stock_quantity.append(no)
smallest_date = min(date_bought)
l=len(date_bought)
i=0
while(i<l):
    info = yf.download(stock_name[i],start =date_bought[i],end = dt.datetime.now().strftime("%Y-%m-%d"),interval= '1d' )
    closingprice = info ['Close']
    opeingprice = info['Open']
    highprice = info['High']
    lowprice = info['Low']
    data.append(closingprice)
    data1.append(opeingprice)
    data2.append(highprice)
    data3.append(lowprice)
    i+=1
df = pd.DataFrame(data)
df.to_csv("Closing_Price.csv",mode='w+',index = False,header=False)
df = pd.DataFrame(data1)
df.to_csv("Opening_Price.csv",mode='w+',index = False,header=False)
df = pd.DataFrame(data2)
df.to_csv("High_Price.csv",mode='w+',index = False,header=False)
df = pd.DataFrame(data3)
df.to_csv("Low_Price.csv",mode='w+',index = False,header=False)


def index_data(index,startdate):
    info = yf.download(index,start = startdate ,end = dt.datetime.now().strftime("%Y-%m-%d"),interval= '1d' )
    index_closing = info['Close']
    index_opening = info['Open']
    returns_index = []
    startingprice= (index_opening[0])
    i=1
    while i<len(index_closing) :
        change = ((index_closing[i] - startingprice)/startingprice) *100
        returns_index.append(round(change,2))
        i+=1
    
    return returns_index
def calc_returns(quantity,avgprice,df):
    stockprice = df.values.tolist()
    stockprice=np.array(stockprice)
    stockprice[np.isnan(stockprice)] = 0
    for i in range(len(stockprice)):
        for j in range(len(stockprice[i])):
            if (stockprice[i][j] > 0):
                stockprice[i][j]= stockprice[i][j] * quantity[i]
    currentvaluetotal =[sum(x) for x in zip(*stockprice)]

    investedvalue = np.copy(stockprice)
    for i in range(len(investedvalue)):
        for j in range(len(investedvalue[i])):
            if (investedvalue[i][j] > 0):
                investedvalue[i][j]= avgprice[i] * quantity[i]
    investedvaluetotal =[sum(x) for x in zip(*investedvalue)]

    returns=[]
    i=0
    l1=len(currentvaluetotal)
    while (i < l1):
        data1 = ((currentvaluetotal[i]-investedvaluetotal[i])/(investedvaluetotal[i]))*100
        returns.append(round(data1,2))
        i+=1
    

    return returns
def calc_dates():
    dates=[]
    index1="^NSEI"
    info = yf.download(index1,start=smallest_date,end = dt.datetime.now().strftime("%Y-%m-%d"),interval= '1d')
    dates = info.index
    return dates
def plot_data_candlestick(returns_opening,returns_closing,datetime,start_index,end_index,index1,indicators,returns_high,returns_low):
    data = pd.DataFrame({'x': datetime, 'y': returns_closing})
    data1=pd.DataFrame({'x1':datetime,'y1':returns_opening})
    data2=pd.DataFrame({'y':returns_high})
    data3=pd.DataFrame({'y':returns_low})

    gain = data1.y1<data.y
    loss = data.y<data1.y1
    width = 12 *60 *60 *1000

    
    p = figure(title='Stock Portfolio', x_axis_type="datetime",x_axis_label='Dates', y_axis_label='Returns',width=1000)
    p.xaxis.major_label_orientation = math.pi/4
    p.grid.grid_line_alpha = 0.25

    p.segment(data.x,data2.y,data.x,data3.y,color="black")
    p.vbar(data.x[gain],width,data1.y1[gain],data.y[gain],fill_color="#00ff00",line_color="#00ff00")
    p.vbar(data.x[loss],width,data1.y1[loss],data.y[loss],fill_color="#ff0000",line_color="#ff0000")
    
    for i in index1:
        if i =="NIFTY 50":
            index_name = "^NSEI"
            returns_index= index_data(index_name,smallest_date)
            index_returns_final = returns_index[start_index:end_index+1]
            p.line(datetime,index_returns_final,color="Red",legend_label ="NIFTY 50",line_width=2)

        elif i == "SENSEX":
            index_name ="^BSESN"
            returns_index= index_data(index_name,smallest_date)
            index_returns_final = returns_index[start_index:end_index+1]
            p.line(datetime,index_returns_final,color="Blue",legend_label ="SENSEX",line_width=2)
        elif i == "NIFTY BANK":
            index_name ="^NSEBANK"
            returns_index= index_data(index_name,smallest_date)
            index_returns_final = returns_index[start_index:end_index+1]
            p.line(datetime,index_returns_final,color="orange",legend_label ="NIFTY BANK",line_width=2)
    for indicator in indicators:
        if indicator == "30 Day SMA":
            data['SMA30'] = data['y'].rolling(30).mean()
            p.line(data.x, data.SMA30,color ="purple",legend_label ="30 DAY SMA",line_width=2)
        elif indicator == "100 Day SMA":
            data['SMA100'] = data['y'].rolling(100).mean()
            p.line(data.x,data.SMA100,color ="purple",legend_label ="100 DAY SMA",line_width=2)
        elif indicator == "Linear Regression Line":
            par =np.polyfit(range(len(data['x'].values)),data.y.values,1,full=True)
            slope =par[0][0]
            intercept =par[0][1]
            y_pred= [slope * i + intercept for i in range (len(data['x'].values))]
            p.segment(data.x.iloc[0],y_pred[0],data.x.iloc[-1],y_pred[-1], legend_label="linear Progression", color="red")
            p.legend.click_policy ="hide"
    p.legend.location ="top_left"
    
    show(p)
    return p
def plot_data_linegraph(returns,datetime,start_index,end_index,index1,indicators):
    p = figure(title='Line Graph', x_axis_type="datetime",x_axis_label='Dates', y_axis_label='Returns')
    p.xaxis.major_label_orientation = math.pi/4
    p.grid.grid_line_alpha = 0.25
    p.line(datetime,returns, line_width=2,color="Green",legend_label ="Returns Portfolio")
    for i in index1:
        if i =="NIFTY 50":
            index_name = "^NSEI"
            returns_index= index_data(index_name,smallest_date)
            index_returns_final = returns_index[start_index:end_index+1]
            p.line(datetime,index_returns_final,color="Red",legend_label ="NIFTY 50",line_width=2)

        elif i == "SENSEX":
            index_name ="^BSESN"
            returns_index= index_data(index_name,smallest_date)
            index_returns_final = returns_index[start_index:end_index+1]
            p.line(datetime,index_returns_final,color="Blue",legend_label ="SENSEX",line_width=2)
        elif i == "NIFTY BANK":
            index_name ="^NSEBANK"
            returns_index= index_data(index_name,smallest_date)
            index_returns_final = returns_index[start_index:end_index+1]
            p.line(datetime,index_returns_final,color="orange",legend_label ="NIFTY BANK",line_width=2)
    data = pd.DataFrame({'x': datetime, 'y': returns})
    for indicator in indicators:
        if indicator == "30 Day SMA":
            data['SMA30'] = data['y'].rolling(30).mean()
            p.line(data.x, data.SMA30,color ="purple",legend_label ="30 DAY SMA",line_width=2)
        elif indicator == "100 Day SMA":
            data['SMA100'] = data['y'].rolling(100).mean()
            p.line(data.x,data.SMA100,color ="purple",legend_label ="100 DAY SMA",line_width=2)
        elif indicator == "Linear Regression Line":
            par =np.polyfit(range(len(data['x'].values)),data.y.values,1,full=True)
            slope =par[0][0]
            intercept =par[0][1]
            y_pred= [slope * i + intercept for i in range (len(data['x'].values))]
            p.segment(data.x.iloc[0],y_pred[0],data.x.iloc[-1],y_pred[-1], legend_label="linear Progression", color="red")
            p.legend.click_policy ="hide"
    p.legend.location ="top_left"
    return p
def on_button_click(choice,index1,startdate,enddate,indicators,change):
    stock_dates=[]
    stock_dates = calc_dates()
    df = pd.DataFrame({'date': stock_dates})
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    start_index = df.index.get_loc(startdate)
    end_index = df.index.get_loc(enddate)
    stock_dates_final = stock_dates[start_index:end_index+1]

    df = pd.read_csv("Opening_Price.csv",header=None)
    stock_returns_opening = calc_returns(stock_quantity,stock_avgprice,df)
    stock_returns_final_opening = stock_returns_opening[start_index:end_index+1]

    df = pd.read_csv("Closing_Price.csv",header=None)
    stock_returns_closing = calc_returns(stock_quantity,stock_avgprice,df)
    stock_returns_final_closing = stock_returns_closing[start_index:end_index+1]

    df = pd.read_csv("High_Price.csv",header=None)
    stock_returns_high= calc_returns(stock_quantity,stock_avgprice,df)
    stock_returns_final_high = stock_returns_high[start_index:end_index+1]

    df = pd.read_csv("Low_Price.csv",header=None)
    stock_returns_low = calc_returns(stock_quantity,stock_avgprice,df)
    stock_returns_final_low = stock_returns_low[start_index:end_index+1]
    if choice == 'Candlestick Graph':
        p1 = plot_data_candlestick(stock_returns_final_opening,stock_returns_final_closing,stock_dates_final,start_index,end_index,index1,indicators,stock_returns_final_high,stock_returns_final_low)
        show(p1)
    else:
        p1 = plot_data_linegraph(stock_returns_final_closing,stock_dates_final,start_index,end_index,index1,indicators)
        show(p1)
    curdoc().clear()
    curdoc().add_root(layout)
    curdoc().add_root(column(p1))

change = 0                 
graph_choice = Select(title="Graph Options",options=['Candlestick Graph','Line Graph'])
index_choice = MultiChoice(title="Index Funds",options=["NIFTY 50","SENSEX","NIFTY BANK"])
date_picker_from = DatePicker(title="Start Date", value = smallest_date,min_date = smallest_date,max_date=dt.datetime.now().strftime("%Y-%m-%d"))
date_picker_to = DatePicker(title = "End Date",value=dt.datetime.now().strftime("%Y-%m-%d"), min_date = smallest_date,max_date=dt.datetime.now().strftime("%Y-%m-%d"))
indicator_choice= MultiChoice(title="Indicators",options =["30 Day SMA","100 Day SMA","Linear Regression Line"])
load_button = Button(label = "Load Data",button_type="success")
load_button.on_click(lambda: on_button_click(graph_choice.value,index_choice.value,date_picker_from.value,date_picker_to.value,indicator_choice.value,change))
layout = column(graph_choice,index_choice,date_picker_from,date_picker_to,indicator_choice,load_button)
curdoc().clear()
curdoc().add_root(layout)



