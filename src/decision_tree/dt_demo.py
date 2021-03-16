import pandas as pd
import talib  # Technical Analysis Library (Download both the TA-Lib itself & the Python wrapper)
import numpy as np
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
    print(list(data.columns),data.shape,type(data), type(data['close'].values))


    # Simple Moving Average
    sma = talib.EMA(data['close'].values, timeperiod)[timeperiod:]

    ## Relative Strength Index: https://www.investopedia.com/terms/r/rsi.asp
    rsi = talib.RSI(data['close'].values, timeperiod)[timeperiod:]

    ## Beta Coefficient: https://www.investopedia.com/terms/b/beta.asp
    mfi = talib.MFI(data['high'].values, data['low'].values, data['close'].values, data['Volume'].values, timeperiod)[timeperiod:]

    # Merge multiple feature `Series` into a single Pandas `DataFrame`.
    X = pd.concat([pd.Series(sma),  pd.Series(rsi), pd.Series(mfi)], axis=1)


    # Why am I using `.reset_index`? :
    # https://stackoverflow.com/questions/18548370/pandas-can-only-compare-identically-labeled-dataframe-objects-error
    #y = data['close'][1:].reset_index(drop=True) > data['close'][:-1].reset_index(drop=True)
    #y = y[(timeperiod - 1):]

    y = data['close'][1:].reset_index(drop=True) > data['close'][:-1].reset_index(drop=True)
    y2 = y ##copy

    y = y[(timeperiod - 1):]


    ### Checks if any of the next 5 closes is greater than current close
    #yn = y.to_numpy()
    yn1 = np.where(data['close'].shift(-1) > data['close'], 1, 0)
    yn2 = np.where(data['close'].shift(-2) > data['close'], 1, 0)
    yn3 = np.where(data['close'].shift(-3) > data['close'], 1, 0)
    yn4 = np.where(data['close'].shift(-4) > data['close'], 1, 0)
    yn5 = np.where(data['close'].shift(-5) > data['close'], 1, 0)

    yn1 = yn1[(timeperiod - 1):]
    yn2 = yn2[(timeperiod - 1):]
    yn3 = yn3[(timeperiod - 1):]

    for i in range(len(yn3)):
        y2[i] = yn1[i] or yn2[i] or yn3[i]

    y2= y2[(timeperiod -1):]
    y = y2
    #print('yn',yn1[10:20],yn2[10:20],yn3[10:20], 'y2\n',y2[10:20])

    #print('shapes',X.shape, y2.shape, y.shape)

    #print('y',data['close'][1:].reset_index(drop=True), data['close'][:-1].reset_index(drop=True))

    assert (X.shape[0] == y.shape[0])
    return X, y


if __name__ == '__main__':
    # './rev_Binance_BTCUSDT_d.csv': 1300 days worth data (time unit = day)
    X, y = gen_X_y(csv_path='./csvs/rev_Binance_BTCUSDT_d.csv', timeperiod=30)

    # Split train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    dt_clf = tree.DecisionTreeClassifier().fit(X_train, y_train)

    # Predict class labels
    y_test_pred = dt_clf.predict(X_test)
    # Predict probability for each class label (Could be used as `confidence` measure)
    # dt_clf.predict_proba(X_test)

    print('classification accuracy: ', accuracy_score(y_test, y_test_pred))
