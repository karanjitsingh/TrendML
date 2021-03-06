import pathlib
import os
from datetime import datetime, timezone
import requests
import json
import time
from collections import namedtuple
import numpy as np

index_cache = {}

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


def _getInterval(timeframe):
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



def _requestAndDump(url, filename):
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


def _scrape(timeframe, toTime: datetime, totalCandles, symbol = "BTCUSDT"):
    global index_cache
    if index_cache.has_key(symbol):
        index_cache.pop(symbol)

    global _log

    if timeframe not in supported_timeframes:
        raise "Unsupported timeframe " + str(timeframe)

    interval = _getInterval(timeframe)
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

        index['index'].append([first.open_time, last.close_time, len(result)])
        _dumpObject(result, os.path.join(store, str(first.open_time) + ".json"))

        # prevent request rate limit exceeded 
        time.sleep(0.1)


    _dumpObject(index, os.path.join(store, "index.json"))



def _scrapeLatest(timeframe, totalCandles, symbol = "BTCUSDT"):
    return _scrape(timeframe, datetime.now(tz=timezone.utc), totalCandles, symbol)

def _printlog(msg):
    print(msg)
    _log.write(str(msg) + "\n")

def _verifyChunk(symbol, timeframe, chunkid):
    datafolder = os.path.join(__datadir, "store", symbol, timeframe)
    path = os.path.join(datafolder, str(chunkid) + ".json")
    return os.path.exists(path)
    

def _readChunk(symbol, timeframe, chunkid):
    datafolder = os.path.join(__datadir, "store", symbol, timeframe)
    path = os.path.join(datafolder, str(chunkid) + ".json")

    if not os.path.exists(path)
        raise Exception("Chunk " + str(chunkid) + " does not exist in " + symbol + "." + timeframe)

    with open(path, 'r') as file:
        return json.loads(file.read())

_log = None


def _getNearestOpen(chunk, start):
    i=0
    while(chunk[i][0] <= start):
        i+=1

    if (i == len(chunk) and start != chunk[i-1])
        raise Exception("No open found")
    
    return i-1

class Series:

    def __init__(indicators, start, buffer, symbol, timeframe):

        symbol = upper(symbol)
        timeframe = lower(timeframe)

        datafolder = os.path.join(__datadir, "store", symbol, timeframe)
        index = os.path.join(datafolder, "index.json")
        
        # check index exists
        if not os.path.exists(index):
            raise Exception("Data store not found for " + str(symbol) + "." + str(timeframe))

        # read index
        with open(index, 'r') as file:
            index = json.loads(file.read())

        # sanity check the index
        if lower(index['time']) != timeframe or upper(index['symbol']) != symbol or (index['index']) == 0:
            raise Exception("Sanity check fail")

        for r in range(len(index['index'])):
            if not os.path.exists(os.path.join(datafolder, str(index['index'][r][0]) + ".json"))
                raise Exception("Sanity check fail")

        self._start = start
        self._buffer = buffer
        self._store = datafolder
        self._buffer_interval = buffer * _getInterval(timeframe)
        
        self._buffer_interval = _getInterval(timeframe) * buffer
        self._indicators = indicators
        self._chunk = None

        index = np.array(index['index'])
        open = index[:, 0]
        close = index[:, 1]

        pointer = None
        pointer_time = start - self._buffer_interval
        
        if pointer_time  < open[0]:
            raise Exception("start - buffer is less than first available candle")

        # find closes candle start
        i = 0
        while i != len(open) and open[i] <= pointer_time 
            i+=1

        if i == len(open):
            if(start > close[i-1] - _getInterval(timeframe)):
                raise Exception("start is higher last available candle")
            else:
                self._chunk = _readChunk(symbol, timeframe, open[i-1])
                pointer = (open[i-1], _getNearestOpen(chunk, pointer_time))
                
            pointer = (open[i-1], 0)
        else:
            self._chunk = _readChunk(symbol, timeframe, open[i-1])
            pointer = (open[i-1], _getNearestOpen(chunk, pointer_time))    

        if pointer == None:
            raise Exception("Missed something")
            
        self._pointer = pointer


    def get(length):
