import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unicodedata import normalize
from datetime import datetime
import requests
import math
import os
import json
import time

def main(parameters, ohlc_call):
    ohlc = get_ohlc(ohlc_call)
    df = get_indicators(ohlc, parameters)
    write_directory_output('print',df[-6:], 'df2csv')
    return

def get_ohlc(ohlc_call):
    url = 'https://chartdata.sfox.com/candlesticks'
    pair = ohlc_call['ticker'] + "usd"
    period = ohlc_call['period']
    startTime = ohlc_call['startStamp']
    endTime = ohlc_call['endStamp']

    url = "{}?endTime={}&pair={}&period={}&startTime={}".format(url, endTime, pair, period, startTime)
    json = requests.get(url).json()

    df = pd.DataFrame(json)
    df['start_time'] = [datetime.fromtimestamp(t) for t in df['start_time']]
    del df['volume']
    del df['pair']
    del df['candle_period']
    del df['vwap']
    del df['trades']
    df = df.set_axis(['open', 'high', 'low', 'close','time'], axis=1)
    df.iloc[:,0:4] = df.iloc[:,0:4].apply(pd.to_numeric)
    df['time'] = [t for t in df['time']]
    #df.index = df['time']
    return df

def get_indicators(df, parameters):
    df = add_emas(df, parameters)
    df = add_stochRSI(df, parameters)
    df = add_atr(df, parameters)
    return df
    #df[['prime']]= [int(i**(2)) for i in df['prime']]
    #.loc[row,column]
    #df = expand_df(df)
    #print(df.iloc[-2,-1])
    #print(df.iloc[-1,-1])


    #Trim rows,  lines are equivalent...[a:b] == ...[a]...[b+1]
    #a = df[1:6]
    #print(a)
    #b = df.loc[df.index[4]:df.index[7]]
    #print(b)

    #Concatenation or Joining
    #c = pd.concat([a,b],axis=0,join='inner').drop_duplicates()
    #print(c)
    #d = a.append(b).drop_duplicates()
    #print(d)

    #Intersection
    #c = pd.concat([a,b],axis=1,join='inner')
    #print(c)

    #Rearrangement columns
    #columns = df.columns.to_list()
    #df.columns = columns[2:] + columns[0:2]
    #print(df)


##Add source column in df["dif"]
def add_emas(df, parameters):
    emasrc = parameters['EMA']['emasrc']
    ema1 = parameters['EMA']['ema1']
    ema2 = parameters['EMA']['ema2']
    ema3 = parameters['EMA']['ema3']
    col1 = 'ema'+str(ema1)
    col2 = 'ema'+str(ema2)
    col3 = 'ema'+str(ema3)
    df[col1] =  df[emasrc].ewm(span=ema1, adjust=False).mean()
    df[col2] =  df[emasrc].ewm(span=ema2, adjust=False).mean()
    df[col3] =  df[emasrc].ewm(span=ema3, adjust=False).mean()
    return df

def add_stochRSI(df, parameters):
    df = rsi(df, parameters)
    df = stoch(df, parameters)
    df = k_d(df, parameters)
    return df

#https://stackoverflow.com/questions/57006437/calculate-rsi-indicator-from-pandas-dataframe
def rsi(df, parameters):
    src = parameters['RSI']['rsisrc']
    n = parameters['RSI']['rsiLength']
    col = 'rsi'

    RSIdf = df.copy()
    RSIdf['change'] = RSIdf[src].diff()
    RSIdf['gain'] = RSIdf.change.mask(RSIdf.change < 0, 0.0)
    RSIdf['loss'] = -RSIdf.change.mask(RSIdf.change > 0, -0.0)
    RSIdf['avg_gain'] = rma(RSIdf.gain[n+1:].to_numpy(), n, np.nansum(RSIdf.gain.to_numpy()[:n+1])/n)
    RSIdf['avg_loss'] = rma(RSIdf.loss[n+1:].to_numpy(), n, np.nansum(RSIdf.loss.to_numpy()[:n+1])/n)
    RSIdf['rs'] = RSIdf.avg_gain / RSIdf.avg_loss
    RSIdf[col] = 100 - (100 / (1 + RSIdf.rs))
    df[col] = RSIdf[col]
    return df

#https://stackoverflow.com/questions/57006437/calculate-rsi-indicator-from-pandas-dataframe
def rma(x, n, y0):
    a = (n-1) / n
    ak = a**np.arange(len(x)-1, -1, -1)
    return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x)+1)]
    #df[['pos']]= [int(i) for i in df['pos']]

#STOCH scale to 100 here
def stoch(df, parameters):
    l = parameters['RSI']['stochLength']
    rsi = df['rsi'].to_list()

    lowest_low = []
    highest_high = []
    for i in range(len(rsi)):
        s = i - l
        if s < 0:
            s = 0
        lowest_low.append(min(rsi[s:i+1]))
        highest_high.append(max(rsi[s:i+1]))


    STOCHdf = df.copy()
    STOCHdf['lowest_low'] = lowest_low
    STOCHdf['highest_high'] = highest_high
    STOCHdf['stoch'] = 100*(STOCHdf['rsi']-STOCHdf['lowest_low'])/(STOCHdf['highest_high']-STOCHdf['lowest_low'])
    df['stoch'] = STOCHdf['stoch']
    return df

def k_d(df, parameters):
    kPeriod = parameters['RSI']['k']
    dPeriod = parameters['RSI']['d']

    df['k'] = df['rsi'].rolling(window=kPeriod).mean()
    df['d'] = df['rsi'].rolling(window=dPeriod).mean()
    return df

def add_atr(df, parameters):
    l = parameters['ATR']['atrLength']
    col = 'atr'

    data = df.copy()
    high = data['high']
    low = data['low']
    close = data['close']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)

    df[col] = tr.rolling(l).sum()/l
    return df

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
    IO = path + "/" + "backtest"
    if not os.path.exists(IO):
        os.mkdir(IO)
    os.chdir(IO)
    return home_dir

if __name__ == '__main__':
    parameters = {
        'EMA': {
            'emasrc':'close',
            'ema1': 8,
            'ema2': 14,
            'ema3': 50,
        },
        'RSI': {
            'rsisrc':'close',
            'k': 3,
            'd': 3,
            'rsiLength': 14,
            'stochLength': 14,
            'upperBand': 90,
            'lowerband':19,
        },
        'ATR': {
            'atrLength': 14
        }
    }

    startTime = datetime(2021, 1, 1, 0, 0, 0)
    endTime = datetime.now()

    ohlc_call = {
        'ticker': 'btc',
        'startTime': str(startTime),
        'startStamp': str(datetime.timestamp(startTime)).split('.')[0],
        'endTime': str(endTime),
        'endStamp': str(datetime.timestamp(endTime)).split('.')[0],
        'period': str(3600)
    }
    main(parameters, ohlc_call)
