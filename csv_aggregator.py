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

def main():
    ohlc = pd.read_csv("./ohlc/binanceBTCUSDT5m2017-08-16 00:00:00.csv")
    titles = [
        'binanceBTCUSDT5m2018-08-16 00:00:00',
        'binanceBTCUSDT5m2019-08-16 00:00:00',
        'binanceBTCUSDT5m2020-08-16 00:00:00'
    ]
    for title in titles:
        new_ohlc = pd.read_csv("./ohlc/"+title+".csv")
        ohlc = pd.concat([ohlc, new_ohlc],axis=0,join='inner').drop_duplicates()

    write_directory_output("AGGREGATEbinanceBTCUSDT5m2017-08-16 00:00:00.csv", ohlc, 'df2csv')
    return

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
    main()
