import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unicodedata import normalize
from datetime import datetime
import backoff
import requests
import math
import os
import json
import time

def main(parameters):
    df = get_historic_with_indicators(parameters)
    df = add_signals(df, parameters)
    df = add_position(df, parameters)
    df = add_metrics(df)
    write_directory_output("example", df, 'df2csv')
    return

def get_historic_with_indicators(parameters):
    i = parameters['interval']
    start = str(parameters['startTime'])
    print(start)
    end = str(parameters['endTime'])
    print(end)
    title = "./added_indicators/binanceBTCUSDT"+i+"2017-08-16 00:00:00.csv"
    df = pd.read_csv(title)
    df.iloc[:,1:] = df.iloc[:,1:].apply(pd.to_numeric)
    df.index = df['closeTime']
    df = df.loc[start:end]
    return df
#crossup = k[1] >= d[1] and k[2] <= d[2] and k <= 60 and k >= 10
#barbuy = crossup and ema1 > ema2 and ema2 > ema3 and close > ema1
#crossdown = k[1] <= d[1] and k[2] >= d[2] and k >= 40 and k <= 95
#barsell = crossdown and ema3 > ema2 and ema2 > ema1 and close < ema1
def add_signals(df, parameters):
    df['change'] = df['close'].pct_change()
    df = add_crossup_signal(df, parameters)
    df = add_crossdown_signal(df, parameters)
    df = add_barbuy_signal(df)
    df = add_barsell_signal(df)
    return df

#crossup = k[1] >= d[1] and k[2] <= d[2] and k <= 60 and k >= 10
def add_crossup_signal(df, parameters):
    upper = parameters['crossupUpper']
    lower = parameters['crossupLower']
    df['crossup'] = False
    for i in range(len(df)):
        cond1 = (df.loc[df.index[i-1], 'k'] >= df.loc[df.index[i-1],'d'])
        cond2 = (df.loc[df.index[i-2],'k'] <= df.loc[df.index[i-2],'d'])
        cond3 = (df.loc[df.index[i],'k'] <= upper)
        cond4 = (df.loc[df.index[i],'k'] >= lower)
        if cond1 and cond2 and cond3 and cond4:
            df.loc[df.index[i],'crossup'] = True
    return df

#crossdown = k[1] <= d[1] and k[2] >= d[2] and k >= 40 and k <= 95
def add_crossdown_signal(df, parameters):
    upper = parameters['crossdownUpper']
    lower = parameters['crossdownLower']
    df['crossdown'] = False
    for i in range(len(df)):
        cond1 = (df.loc[df.index[i-1], 'k'] <= df.loc[df.index[i-1],'d'])
        cond2 = (df.loc[df.index[i-2],'k'] >= df.loc[df.index[i-2],'d'])
        cond3 = (df.loc[df.index[i],'k'] <= upper)
        cond4 = (df.loc[df.index[i],'k'] >= lower)
        if cond1 and cond2 and cond3 and cond4:
            df.loc[df.index[i],'crossdown'] = True
    return df
#barbuy = crossup and close > ema1 and ema1 > ema2 and ema2 > ema3
def add_barbuy_signal(df):
    df['barbuy'] = False
    for i in range(len(df)):
        cond1 = (df.loc[df.index[i], 'crossup'])
        cond2 = (df.loc[df.index[i],'close'] > df.loc[df.index[i],'ema1'])
        cond3 = (df.loc[df.index[i], 'ema1'] > df.loc[df.index[i], 'ema2'])
        cond4 = (df.loc[df.index[i], 'ema2'] > df.loc[df.index[i], 'ema3'])
        if cond1 and cond2 and cond3 and cond4:
            df.loc[df.index[i], 'barbuy'] = True
    return df
#barsell = crossdown and ema3 > ema2 and ema2 > ema1 and ema1 > close
def add_barsell_signal(df):
    df['barsell'] = False
    for i in range(len(df)):
        cond1 = (df.loc[df.index[i], 'crossdown'])
        cond2 = (df.loc[df.index[i],'close'] < df.loc[df.index[i],'ema1'])
        cond3 = (df.loc[df.index[i], 'ema1'] < df.loc[df.index[i], 'ema2'])
        cond4 = (df.loc[df.index[i], 'ema2'] < df.loc[df.index[i], 'ema3'])
        if cond1 and cond2 and cond3 and cond4:
            df.loc[df.index[i], 'barsell'] = True
    return df

def add_position(df, parameters):
    df = add_stoploss_takeprofit(df, parameters)
    df['position'] = 0
    for i in range(len(df)):
        #check for trigger of longloss, close long position
        if (df.loc[df.index[i],'low'] <= df.loc[df.index[i],'longloss']) and (df.loc[df.index[i-1],'position'] >= 0):
            df.loc[df.index[i],'position'] = 0
        #check for trigger of shortloss, close short position
        if (df.loc[df.index[i],'high'] >= df.loc[df.index[i], 'shortloss']) and (df.loc[df.index[i-1],'position'] <= 0):
            df.loc[df.index[i],'position'] = 0
        #check for trigger of longprofit, close long position
        if (df.loc[df.index[i],'high'] >= df.loc[df.index[i],'longprofit']) and (df.loc[df.index[i-1],'position'] >= 0):
            df.loc[df.index[i],'position'] = 0
        #check for trigger of shortprofit, close short position
        if (df.loc[df.index[i],'low'] <= df.loc[df.index[i], 'shortprofit']) and (df.loc[df.index[i-1],'position'] <= 0):
            df.loc[df.index[i],'position'] = 0
        #check for long position
        if df.loc[df.index[i-1],'position'] >= 0:
            if (df.loc[df.index[i],'barbuy'] == True):
                df.loc[df.index[i],'position'] = 1
        #check for short position
        if df.loc[df.index[i-1],'position'] <= 0:
            if (df.loc[df.index[i],'barsell'] == True):
                df.loc[df.index[i],'position'] = 1
        #continue otherwise
        if df.loc[df.index[i-1],'position'] == 1:
            df.loc[df.index[i],'position'] = 1
        if df.loc[df.index[i-1],'position'] == -1:
            df.loc[df.index[i],'position'] = -1
    return df

#longloss = sma(open, 1)
#shortloss = sma(open, 1)
#longloss := barbuy ? close - (atr * atr_loss) : longloss[1]
#shortloss := barsell ? close + (atr * atr_loss) : shortloss[1]
def add_stoploss_takeprofit(df, parameters):
    pMult = parameters['takeProfitAtrMultiplier']
    lMult = parameters['stopLossAtrMultiplier']
    df['longloss'] = df['open'].rolling(window=1).mean()
    df['shortloss'] = df['open'].rolling(window=1).mean()
    df['longprofit'] = df['open'].rolling(window=1).mean()
    df['shortprofit'] = df['open'].rolling(window=1).mean()
    for i in range(len(df)):
        if df.loc[df.index[i],'barbuy'] == True:
            df.loc[df.index[i], 'longloss'] = df.loc[df.index[i],'close'] - (df.loc[df.index[i],'atr']*lMult)
            df.loc[df.index[i], 'longprofit'] = df.loc[df.index[i],'close'] + (df.loc[df.index[i],'atr']*pMult)
        else:
            df.loc[df.index[i],'longloss'] = df.loc[df.index[i-1],'longloss']
            df.loc[df.index[i], 'longprofit'] = df.loc[df.index[i-1], 'longprofit']
        if df.loc[df.index[i],'barsell'] == True:
            df.loc[df.index[i],'shortloss'] = df.loc[df.index[i],'close'] + (df.loc[df.index[i],'atr']*lMult)
            df.loc[df.index[i],'shortprofit'] = df.loc[df.index[i],'close'] - (df.loc[df.index[i],'atr']*pMult)
        else:
            df.loc[df.index[i],'shortloss'] = df.loc[df.index[i-1],'shortloss']
            df.loc[df.index[i],'shortprofit'] = df.loc[df.index[i-1],'shortprofit']
    return df

def add_metrics(df):
    df, net, buyAndHold = net_growth(df)
    print('net: '+str(net))
    print('buyAndHold: '+str(buyAndHold))
    return df

def net_growth(df):
    df['pctchange'] = df['close'].pct_change() + 1
    df['change'] = df['position'] * df['pctchange']
    for i in range(len(df)):
        if df.loc[df.index[i],'change'] == 0.0:
            df.loc[df.index[i],'change'] = 1.0
    print(df['change'])
    df['net'] = np.cumprod(df['change'])
    net = df.loc[df.index[-1],'net']
    print('0, open'+str(df.loc[df.index[0],'open'])+" "+str(df.index[0]))
    print('-1 close'+str(df.loc[df.index[-1],'close'])+" "+str(df.index[-1]))
    buyAndHold = (df.loc[df.index[-1],'close']-df.loc[df.index[0],'open'])/df.loc[df.index[0],'open']
    return df, net, buyAndHold

def write_directory_output(text_name, data, format):
    home_dir = initialize_directory(text_name)
    if format == 'df2csv':
        write_csv(text_name, data)
    if format == 'json':
        write_json(text_name, data)
    os.chdir(home_dir)
    return
# produces csv for each game, columns as team names, then roster for rows
def write_csv(text_name, data):
    path = os.getcwd()
    data.to_csv(text_name+".csv", index=False)
    print("Wrote "+text_name+".csv to to path: "+str(path))
    return
def write_json(text_name, data):
    path = os.getcwd()
    with open(text_name + ".json", 'w') as f: #w or wt?
        json.dump(data, f, indent = 4)
    print("Wrote "+str(text_name)+".json to to path: "+str(path))
    return
#3a creates a folder to write output csvs. Folder uses source web domain and adds date and time (HH:MM), embedding project metadata.  Move to output folder, returns home directory string.
def initialize_directory(text_name):
    path = os.getcwd()
    home_dir = path
    #following line forms output folder name
    IO = path + "/" + "added_conditionals"
    if not os.path.exists(IO):
        os.mkdir(IO)
    os.chdir(IO)
    return home_dir

if __name__ == '__main__':
    parameters = {
        'interval': '1d',
        'startTime': datetime(2021, 5, 15, 17, 0, 0),
        'endTime': datetime(2021, 9, 15, 17, 0, 0),
        'crossupUpper': 60,
        'crossupLower': 10,
        'crossdownUpper': 95,
        'crossdownLower': 40,
        'stopLossAtrMultiplier': 1,
        'takeProfitAtrMultiplier': 1
    }
    main(parameters)
