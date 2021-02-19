import pathlib
import os
from datetime import datetime, timezone
import requests
import json
import time
from collections import namedtuple

kline = namedtuple('kline', ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'num_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore', ])


__datadir = pathlib.Path(__file__).parent.absolute()
__requestRateLimit = 1200/60
__resultLength = 1000


endpoint  = "https://www.binance.com/api/v1/klines?symbol={symbol}&interval={timeframe}&startTime={start}&endTime={end}&limit=2000"

second = 1000
minute = second * 60
hour = minute * 60
day = hour * 24
week = day * 7

supported_timeframes = ["1m", "2m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "3h", "4h", "6h", "8h", "10h", "12h", "1d", "2d", "3d", "1w"]


def getInterval(timeframe):
    period = int(timeframe[0:-1])
    r = 0
    if timeframe[-1] == "m":
        r = period * minute
    elif timeframe[-1] == "h":
        r = period * hour
    elif timeframe[-1] == "d":
        r = period * day
    elif timeframe[-1] == "w":
        r = period * week

    return r


def _createStore(symbol, timeframe):
    store = os.path.join(__datadir, "store", symbol, timeframe)
    os.makedirs(store, exist_ok=True)
    return store



def requestAndDump(url, filename):
    success = True

    request = requests.get(url)
    data = request.content

    if request.status_code is 200:
        with open(filename, "w") as file:
            file.write(data.__str__()[2:-1])                
    else:
        print("Non 200 status code")
        print("Status Code:", request.status_code)
        print(request.headers)
        print(request.content)
        success = False

    return success

def request(url):
    success = True

    request = requests.get(url)

    if request.status_code is not 200:
        print("Non 200 status code")
        print("Status Code:", request.status_code)
        print(request.headers)
        print(request.content)
        success = False

    return [success, request]

def _scrape(start, length, dumpfolder):
    pairs = getDayPairs(start, length)
    
    for i in range(len(pairs)):
        if timeframe == "1m":
            # max number of records returned by the api is 1000 so we will have to request twice for each day
            
            print(pairs[i][2])
            url = endpoint.format(start=pairs[i][0], end=pairs[i][1]-int(day/2), timeframe=timeframe, symbol=symbol)
            
            print(url)


            if requestAndDump(url, os.path.join(dumpfolder, pairs[i][2] + ".1.json")) == True:
                url = endpoint.format(start=pairs[i][1]-int(day/2), end=pairs[i][1], timeframe=timeframe, symbol=symbol)
                print(url,"\n")

                if requestAndDump(url, os.path.join(dumpfolder, pairs[i][2] + ".2.json")) == False:
                    print("Breaking on " + pairs[i][2])
                    break  
            else:      
                print("Breaking on " + pairs[i][2])
                break

        else:
            print(pairs[i][2])
            url = endpoint.format(start=pairs[i][0], end=pairs[i][1], timeframe=timeframe, symbol=symbol)
            
            print(url)

            if requestAndDump(url, os.path.join(dumpfolder, pairs[i][2] + ".json")) == False:
                print("Breaking on " + pairs[i][2])
                break

        
        time.sleep(delay)

def _dumpObject(data, filename):
    with open(filename, "w") as file:
        file.write(json.dumps(data))

def fetch(timeframe, toTime: datetime, totalCandles, symbol = "BTCUSDT"):

    if timeframe not in supported_timeframes:
        raise "Unsupported timeframe " + str(timeframe)

    interval = getInterval(timeframe)
    _range = __resultLength * interval 
    endtick = int(toTime.timestamp() * 1000)
    starttick = endtick - _range
    
    remaining = totalCandles

    index = {
        'time': timeframe,
        'symbol': symbol,
        'index': []
    }

    store = _createStore(symbol, timeframe)

    while remaining > 0:        
        url = endpoint.format(start=starttick, end=endtick, timeframe=timeframe, symbol=symbol)
        print(url)
        success, r = request(url)

        if not success:
            _dumpObject(index, os.path.join(store, "index.json"))
            raise "Request unsuccessful, stopping here"
        
        result = json.loads(r.content)

        if(len(result) != __resultLength):
            _dumpObject(index, os.path.join(store, "index.json"))
            raise "Data length less than expected length, stopping here"


        first = kline(*result[0])
        last = kline(*result[-1])

        remaining = remaining - __resultLength
        endtick = first.open_time
        starttick = endtick - _range

        index['index'].append([first.open_time, last.close_time])
        _dumpObject(result, os.path.join(store, str(first.open_time) + ".json"))

        # prevent request rate limit exceeded 
        time.sleep(0.1)


    _dumpObject(index, os.path.join(store, "index.json"))


def fetchLatest(timeframe, totalCandles, symbol = "BTCUSDT"):
    return fetch(timeframe, datetime.now(tz=timezone.utc), totalCandles, symbol)

x = fetchLatest("1m", 1000, symbol = "BTCUSDT")