from data import datastore
import os
import talib
from inspect import getmembers, isfunction
import numpy as np
import json

# KLine = datastore.KLine


_index_cache = {}

def _getIndex(symbol, timeframe):
    global _index_cache
    key = symbol.upper() + "." + timeframe.lower()
    
    if key in _index_cache:
        return _index_cache[key]

    datafolder = os.path.join(datastore._datadir, "store", symbol, timeframe)
    index = os.path.join(datafolder, "index.json")
    
    # check index exists
    if not os.path.exists(index):
        raise Exception("Data store not found for " + str(symbol) + "." + str(timeframe))

    # read index
    with open(index, 'r') as file:
        index = json.loads(file.read())

    index['index'] = np.array(index['index'])
    _index_cache[key] = index

    return index

class KlineData:
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DataSeries:

    def __init__(self, symbol, timeframe):

        self._symbol = symbol = symbol.upper()
        self._timeframe = timeframe = timeframe.lower()
        self._indicators = {}

        datafolder = os.path.join(datastore._datadir, "store", symbol, timeframe)

        index = _getIndex(symbol, timeframe)

        # sanity check the index
        if index['time'].lower() != timeframe or index['symbol'].upper() != symbol or len(index['index']) == 0:
            raise Exception("Sanity check fail")

        for r in range(len(index['index'])):
            if not os.path.exists(os.path.join(datafolder, str(index['index'][r][0]) + ".json")):
                raise Exception("Sanity check fail")

        self._store = datafolder
        self._indicators = dict()
        self._chunk = None

        self._index = index['index']

        d = np.empty((0,12))

        for i in range(len(index['index'])):
            with open(os.path.join(datafolder, str(index['index'][i][0]) + ".json"), 'r') as chunk:
                chunk = json.loads(chunk.read())
                d = np.vstack((np.array(chunk), d))

        self._data = {
            "_": d.astype('double')
        }

        for i in range(len(datastore.data_columns)):
            self._data[datastore.data_columns[i]] = self._data['_'][:,i]


    def getStartTime(self):
        return self._index[0][0]

    def getEndTime(self):
        return self._index[-1][0]

    def getData(self):
        return self._data
    
    def getIndicators(self):
        return self._indicators

    def addIndicator(self, indicator, *args, **kwargs):
        if indicator not in talib.get_functions():
            raise Exception("Indicator '" + str(indicator) + "' not supported")

        handle = getattr(talib, indicator)

        self._indicators[indicator] = handle(*args, **kwargs)
