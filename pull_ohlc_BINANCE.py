import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unicodedata import normalize
from datetime import datetime
import time
import backoff
import requests
import math
import os
import json

def main(ohlc_call):
    ohlc = clean_ohlc(get_ohlc(ohlc_call, df=[]))

    title = ohlc_call['exchange']+ohlc_call['ticker']+ohlc_call['interval']+ohlc_call['startTime']
    write_directory_output(title, ohlc, 'df2csv')
    return

@backoff.on_exception(backoff.expo, requests.exceptions.RequestException)
def get_ohlc(ohlc_call, df):
    url = ohlc_call['url']
    symbol =  ohlc_call['ticker']
    interval = ohlc_call['interval']
    startTime = ohlc_call['startStamp']
    endTime = ohlc_call['endStamp']

    url = "{}?symbol={}&interval={}&startTime={}".format(url, symbol, interval, startTime)
    ohlc = requests.get(url).json()

    columns = ['openTime','open','high', 'low', 'close','volume','closeTime','quoteAssetVolume','trades','takerBuyBaseVol','takerSellQuoteVol','ignore']
    df = pd.DataFrame(ohlc, columns=columns)

    del df['volume']
    del df['quoteAssetVolume']
    del df['trades']
    del df['takerBuyBaseVol']
    del df['takerSellQuoteVol']
    del df['ignore']
    df = df.set_axis(['openTime','open', 'high', 'low', 'close','closeTime'], axis=1)

    #iterate for full time period
    last_close = df['closeTime'].iloc[-1] +1
    dif = int(ohlc_call['endStamp']) - int(last_close)
    if dif > 3600000:
        ohlc_call['startStamp'] = df['closeTime'].iloc[-1] +1
        df2 = get_ohlc(ohlc_call, df)
        df = pd.concat([df,df2],axis=0,join='inner').drop_duplicates()
    return df

def clean_ohlc(df):
    del df['openTime']
    df.iloc[:,0:4] = df.iloc[:,0:4].apply(pd.to_numeric)
    df = df[['closeTime','open','high','low','close']]
    df['closeTime'] = [datetime.fromtimestamp((t+1)/1000.0) for t in df['closeTime']]
    df.index = df['closeTime']
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
    IO = path + "/" + "ohlc"
    if not os.path.exists(IO):
        os.mkdir(IO)
    os.chdir(IO)
    return home_dir

if __name__ == '__main__':

    startTime = datetime(2020, 8, 16, 0, 0, 0)
    endTime = datetime.now()
    ohlc_call = {
        'exchange': 'binance',
        'url': 'https://api.binance.com/api/v3/klines',
        'ticker': 'BTCUSDT',
        'startTime': str(startTime),
        'startStamp': str(int(datetime.timestamp(startTime)*1000)),
        'endTime': str(endTime),
        'endStamp': str(int(datetime.timestamp(endTime)*1000)),
        'interval': '5m'
    }
    main(ohlc_call)
