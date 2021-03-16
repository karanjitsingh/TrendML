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

    # Why am I using `.reset_index`? :
    # https://stackoverflow.com/questions/18548370/pandas-can-only-compare-identically-labeled-dataframe-objects-error
    y = data['close'][1:].reset_index(drop = True) > data['close'][:-1].reset_index(drop = True)
    y = y[(timeperiod - 1):]

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
