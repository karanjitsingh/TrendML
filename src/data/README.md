# Data

## Download data store

Download data store from https://gtvault-my.sharepoint.com/personal/ksingh323_gatech_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fksingh323%5Fgatech%5Fedu%2FDocuments%2Funtitled%20ml%20project%2Fstore


## Using the DataSeries object

See the sample script below with defined usages for methods on the `DataSeries` object.


```python
>>> from data import series                                         # Import the series module

>>> btc1h = series.DataSeries('BTCUSDT', '1h')                      # Create a series for BTC-USDT pair on 1h candles 

>>> btc1h.getData()                                                 # Fetch data with getData method
{'_': array([[1.50294240e+12, 4.26148000e+03, 4.31362000e+03, ...,
        3.51605030e+01, 1.50952478e+05, 7.88763551e+03],
       [1.50294600e+12, 4.30883000e+03, 4.32869000e+03, ...,
        2.14480710e+01, 9.26082797e+04, 8.03926240e+03],
       [1.50294960e+12, 4.33029000e+03, 4.34545000e+03, ...,
        4.80286100e+00, 2.07953172e+04, 8.04176050e+03],
       ...,
       [1.61376480e+12, 5.49948500e+04, 5.56981600e+04, ...,
        1.85122755e+03, 1.02289449e+08, 0.00000000e+00],
       [1.61376840e+12, 5.53451700e+04, 5.63680000e+04, ...,
        2.54036048e+03, 1.42019526e+08, 0.00000000e+00],
       [1.61377200e+12, 5.56511600e+04, 5.57672900e+04, ...,
        1.48046711e+03, 8.21048732e+07, 0.00000000e+00]]), 'open_time': array([1.5029424e+12, 1.5029460e+12, 1.5029496e+12, ..., 1.6137648e+12,
       1.6137684e+12, 1.6137720e+12]), 'open': array([ 4261.48,  4308.83,  4330.29, ..., 54994.85, 55345.17, 55651.16]), 'high': array([ 4313.62,  4328.69,  4345.45, ..., 55698.16, 56368.  , 55767.29]), 'low': array([ 4261.32,  4291.37,  4309.37, ..., 54800.  , 55304.52, 55130.  ]), 'close': array([ 4308.83,  4315.32,  4324.35, ..., 55348.33, 55651.16, 55426.4 ]), 'volume': array([  47.181009,   23.234916,    7.229691, ..., 3733.048485,
       5011.990364, 3226.954069]), 'close_time': array([1.5029460e+12, 1.5029496e+12, 1.5029532e+12, ..., 1.6137684e+12,
       1.6137720e+12, 1.6137756e+12]), 'quote_asset_volume': array([2.02366138e+05, 1.00304824e+05, 3.12823127e+04, ...,
       2.06292560e+08, 2.80151757e+08, 1.78971150e+08]), 'num_trades': array([1.71000e+02, 1.02000e+02, 3.60000e+01, ..., 1.15558e+05,
       1.46559e+05, 9.95220e+04]), 'taker_buy_base_asset_volume': array([  35.160503,   21.448071,    4.802861, ..., 1851.227548,
       2540.360481, 1480.467108]), 'taker_buy_quote_asset_volume': array([1.50952478e+05, 9.26082797e+04, 2.07953172e+04, ...,
       1.02289449e+08, 1.42019526e+08, 8.21048732e+07]), 'ignore': array([7887.63551305, 8039.26240152, 8041.76049845, ...,    0.        ,
          0.        ,    0.        ])}

>>> btc1h.getData().keys()                                          # getData essentially returns a dictionary of numpy arrays, each column can be accessed with its own keys
dict_keys(['_', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'num_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

>>> btc1h.getData()['close']                                        # Accessing the data
array([ 4308.83,  4315.32,  4324.35, ..., 55348.33, 55651.16, 55426.4 ])


>>> btc1h.addIndicator('RSI', btc1h.getData()['close'])             # Adding indicators to the data series

>>> btc1h.getIndicators()                                           # getIndicators returns a dictionary of fetched indicators
{'RSI': array([        nan,         nan,         nan, ..., 78.725809  ,
       80.21688503, 75.96144231])}

>>> btc1h.getIndicators()['RSI']                                    # See TALib API docs to find the individual parameters for each indicator
array([        nan,         nan,         nan, ..., 78.725809  ,
       80.21688503, 75.96144231])


```


## Data Sources
* Cryptocurrency (Binance Exchange Data)
  (day/hour/minute)
  https://www.cryptodatadownload.com/data/binance/

## TALib API Docs:
https://github.com/mrjbq7/ta-lib/tree/master/docs/func_groups