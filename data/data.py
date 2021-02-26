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

    if request.status_code == 200:
        with open(filename, "w") as file:
            file.write(data.__str__()[2:-1])                
    else:
        _printlog("Non 200 status code")
        _printlog("Status Code:", request.status_code)
        _printlog(request.headers)
        _printlog(request.content)
        success = False

    return success

def request(url):
    success = True

    request = requests.get(url)

    if request.status_code !=  200:
        _printlog("Non 200 status code")
        _printlog(request.status_code)
        _printlog(request.headers)
        _printlog(request.content)
        success = False

    return [success, request]

def _dumpObject(data, filename):
    with open(filename, "w") as file:
        file.write(json.dumps(data))

def _gapAnalysis(data, start, end, interval):
    _printlog("Performing gap analysis:")
    
    if(kline(*data[0]).open_time != start):
        _printlog("First candle deson't match")
    if(kline(*data[-1]).close_time != end and kline(*data[-1]).close_time != end - 1):
        _printlog("Last candle deson't match")
    
    for i in range(0, len(data) - 1):
        curr = kline(*data[i])
        next = kline(*data[i+1])

        if(curr.close_time + 1 != next.open_time):
            _printlog("Gap on close: " + str(curr.close_time) + " and open: " + str(next.open_time) + " at " + str(i) + ", difference: " + str(next.open_time - curr.close_time) + ", ratio: " + str((next.open_time - curr.close_time)/interval) + " datetime: " + str(datetime.utcfromtimestamp(curr.close_time/1000)))


def fetch(timeframe, toTime: datetime, totalCandles, symbol = "BTCUSDT"):
    global _log

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
    _log = open(os.path.join(store, timeframe + "." + symbol + ".log"), "w")

    while remaining > 0:        
        url = endpoint.format(start=starttick, end=endtick-1, timeframe=timeframe, symbol=symbol)
        _printlog(url + "    Time: " + str(datetime.utcfromtimestamp(starttick/1000)))
        success, r = request(url)

        if not success:
            _dumpObject(index, os.path.join(store, "index.json"))
            raise "Request unsuccessful, stopping here"
        
        result = json.loads(r.content)

        if(len(result) != __resultLength):
            _dumpObject(index, os.path.join(store, "index.json"))
            _printlog("Result length less than expected")
            _gapAnalysis(result, starttick, endtick,interval)


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

def _printlog(msg):
    print(msg)
    _log.write(str(msg) + "\n")

_log = None