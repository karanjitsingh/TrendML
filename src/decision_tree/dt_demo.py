import numpy as np
import pandas as pd
import talib # Technical Analysis Library (Download both the TA-Lib itself & the Python wrapper)

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def gen_X_y(csv_path: str, timeperiod: int):
    """Calculate technical indicators & truth labels

    Args:
    csv_path: path to .csv file that contains historic price data
    timeperiod: # of timeperiods to be used for calculating technical indicators

    Returns:
        (X, y): (feature matrix, truth labels)
    """
    # Read .csv file into a pandas dataframe
    data = pd.read_csv(csv_path)
    # print(list(data.columns))
    
    # Simple Moving Average
    sma = talib.SMA(data['close'], timeperiod)[timeperiod:]

    ## Relative Strength Index: https://www.investopedia.com/terms/r/rsi.asp
    rsi = talib.RSI(data['close'], timeperiod)[timeperiod:]

    ## Beta Coefficient: https://www.investopedia.com/terms/b/beta.asp
    beta = talib.BETA(data['high'], data['low'], timeperiod)[timeperiod:]

    # Merge multiple feature `Series` into a single Pandas `DataFrame`.
    X = pd.concat([sma, rsi, beta], axis=1)

    # `.values` of a Pandas dataframe refers to its NumPy array representation.
    y = data['close'][1:].values > data['close'][:-1].values
    y = y[(timeperiod - 1):]

    # Another option of generating truth labels `y`:
    #   Compare with previous 5 timeframes, instead of simply observing immediate up/down change
    #   Create a new table with price info of 6 consecutive timeframes in one row.
    prev5 = pd.concat([
            # pivot timeframe
            data['close'][5:].reset_index(drop = True),
            # previous 5 timeframes
            data['close'][4:-1].reset_index(drop = True), # 1 frame ago
            data['close'][3:-2].reset_index(drop = True), # 2 frame ago
            data['close'][2:-3].reset_index(drop = True), # 3 frame ago
            data['close'][1:-4].reset_index(drop = True), # 4 frame ago
            data['close'][:-5].reset_index(drop = True),  # 5 frame ago
        ],
        axis = 1
    )
    prev5.columns = ['pivot', '1_ago', '2_ago', '3_ago', '4_ago', '5_ago'] # rename columns
    prev5 = prev5[(timeperiod - 5):]
    # print(prev5.shape)

    # Example of generating `y` using `prev5`:
    #     Take minimum of previous 5 days, to check whether the pivot day exceeds the minimum.
    y = prev5['pivot'] > np.amin(prev5[['1_ago', '2_ago', '3_ago', '4_ago', '5_ago']], axis = 1)

    assert(X.shape[0] == y.shape[0])
    return X, y

if __name__ == '__main__':
    # './rev_Binance_BTCUSDT_d.csv': 1300 days worth data (time unit = day)
    X, y = gen_X_y(csv_path = './csvs/rev_Binance_BTCUSDT_d.csv', timeperiod = 30)
    
    # Split train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    dt_clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
    
    # Predict class labels
    y_test_pred = dt_clf.predict(X_test)
    # Predict probability for each class label (Could be used as `confidence` measure)
    # dt_clf.predict_proba(X_test)

    print('classification accuracy: ', accuracy_score(y_test, y_test_pred))
