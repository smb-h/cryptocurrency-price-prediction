import json
import requests
import pandas as pd


url = "https://min-api.cryptocompare.com/data/"
api_key = "fbd094f08b216fbd27e436c1134c34aeac9cf6b823f425ce4e67c43d322b7081"


def get_top100():
    endpoint = "top/mktcapfull"
    res = requests.get(url + endpoint + '?tsym=USD&limit=100&api_key=' + api_key)
    df = pd.DataFrame(json.loads(res.content)['Data'])
    names = []
    for i in df.get("CoinInfo"):
        names.append(i.get("Name"))
    return names


def crawl():
    endpoint = "v2/histoday"
    # endpoint = "histoday"
    coin = "BTC"
    res = requests.get(url + endpoint + '?fsym=' + coin + '&tsym=USD&limit=500&api_key=' + api_key)

    # print(res.content)
    records = pd.DataFrame(json.loads(res.content)['Data']['Data'])
    records = records.set_index('time')
    records.index = pd.to_datetime(records.index, unit='s')

    # print(records.head(5))
    # sync code to work with these
    # high - low - open - volumefrom - volumeto - close
    del records["conversionType"]
    del records["conversionSymbol"]

    return records




