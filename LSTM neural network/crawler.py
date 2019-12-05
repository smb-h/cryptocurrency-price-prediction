import json
import requests
import pandas as pd


def crawl():
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    api_key = "fbd094f08b216fbd27e436c1134c34aeac9cf6b823f425ce4e67c43d322b7081"
    res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=500&api_key=' + api_key)

    # print(res.content)
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')

    # print(hist.head(5))

    return hist

