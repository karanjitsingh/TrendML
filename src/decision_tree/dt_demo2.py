import pandas as pd
import talib  # Technical Analysis Library (Download both the TA-Lib itself & the Python wrapper)
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os



def gen_X_y(csv_path: str, timeperiod: int, indicator_list:list, long:bool):
    """Calculate technical indicators & truth labels

  Args:
  csv_path: path to .csv file that contains historic price data
  timeperiod: # of timeperiods to be used for calculating technical indicators
  indicator_list: list of indicator indices to be used from indicator_sets_ref
  Long: 1 for long(buy) and 0 for short(sell)

  Returns:
      (X, y): (feature matrix, truth labels)
  """
    # Read .csv file into a pandas dataframe
    data = pd.read_csv(csv_path)
    print(list(data.columns),data.shape,type(data), type(data['close'].values))


    # Simple Moving Average
    sma1 = talib.EMA(data['close'].values, 21)[timeperiod:]
    sma2 = talib.EMA(data['close'].values, 50)[timeperiod:]

    # Exponential Moving Average
    ema1 = talib.EMA(data['close'].values, 21)[timeperiod:]
    ema2 = talib.EMA(data['close'].values, 50)[timeperiod:]



    ## Relative Strength Index: https://www.investopedia.com/terms/r/rsi.asp
    #rsi = talib.RSI(data['close'].values, timeperiod)[timeperiod:]
    rsi = talib.RSI(data['close'].values, 14)[timeperiod:]

    ## Beta Coefficient: https://www.investopedia.com/terms/b/beta.asp
    #beta = talib.BETA(data['high'].values, data['low'].values, timeperiod)[timeperiod:]

    ## MFI: https://www.investopedia.com/terms/m/mfi.asp
    #mfi = talib.MFI(data['high'].values, data['low'].values, data['close'].values, data['Volume'].values, timeperiod)[timeperiod:]
    mfi = talib.MFI(data['high'].values, data['low'].values, data['close'].values, data['Volume'].values, 10)[timeperiod:]

    ## ADX: https://www.investopedia.com/terms/a/adx.asp
    adx = talib.ADX(data['high'].values, data['low'].values, data['close'].values, 14)[timeperiod:]

    ## ATR: https://www.investopedia.com/terms/a/atr.asp
    atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, 14)[timeperiod:]


    indicator_sets = {1: sma1, 2: sma2, 3: ema1, 4: ema2, 5: rsi, 6: mfi, 7:adx, 8:atr}

    # Merge multiple feature `Series` into a single Pandas `DataFrame`.
    #X = pd.concat([pd.Series(ema),  pd.Series(rsi), pd.Series(mfi)], axis=1)
    X = pd.concat([pd.Series(indicator_sets[indicator_list[0]]),
                   pd.Series(indicator_sets[indicator_list[1]]),
                   pd.Series(indicator_sets[indicator_list[2]])], axis=1)


    # Why am I using `.reset_index`? :
    # https://stackoverflow.com/questions/18548370/pandas-can-only-compare-identically-labeled-dataframe-objects-error
    #y = data['close'][1:].reset_index(drop=True) > data['close'][:-1].reset_index(drop=True)
    #y = y[(timeperiod - 1):]

    y = data['close'][1:].reset_index(drop=True) > data['close'][:-1].reset_index(drop=True)
    y2 = y ##copy

    y = y[(timeperiod - 1):]


    ### Checks if any of the next 5 closes is greater than current close by 5% or more
    #yn = y.to_numpy()
    if long:
        yn1 = np.where(data['close'].shift(-1) > 1.05*data['close'], 1, 0)
        yn2 = np.where(data['close'].shift(-2) > 1.05*data['close'], 1, 0)
        yn3 = np.where(data['close'].shift(-3) > 1.05*data['close'], 1, 0)
        yn4 = np.where(data['close'].shift(-4) > 1.05*data['close'], 1, 0)
        yn5 = np.where(data['close'].shift(-5) > 1.05*data['close'], 1, 0)

        yn1 = yn1[(timeperiod - 1):]
        yn2 = yn2[(timeperiod - 1):]
        yn3 = yn3[(timeperiod - 1):]
        yn4 = yn4[(timeperiod - 1):]
        yn5 = yn5[(timeperiod - 1):]

        print('prices',y,'\n',yn1,'\n',yn2,'\n',yn3,'\n',yn4,'\n',yn5,'\n')
        for i in range(len(yn3)):
            y2[i] = yn1[i] or yn2[i] or yn3[i] or yn4[i] or yn5[i]

        y2= y2[(timeperiod -1):]
        y = y2

    else:
        perc = 1 - 0.05
        yn1 = np.where(data['close'].shift(-1) < perc * data['close'], 1, 0)
        yn2 = np.where(data['close'].shift(-2) < perc * data['close'], 1, 0)
        yn3 = np.where(data['close'].shift(-3) < perc * data['close'], 1, 0)
        yn4 = np.where(data['close'].shift(-4) < perc * data['close'], 1, 0)
        yn5 = np.where(data['close'].shift(-5) < perc * data['close'], 1, 0)

        yn1 = yn1[(timeperiod - 1):]
        yn2 = yn2[(timeperiod - 1):]
        yn3 = yn3[(timeperiod - 1):]
        yn4 = yn4[(timeperiod - 1):]
        yn5 = yn5[(timeperiod - 1):]

        print('prices', '\n', yn1[0:50],yn5[40:80])
        for i in range(len(yn3)):
            y2[i] = yn1[i] or yn2[i] or yn3[i] or yn4[i] or yn5[i]

        y2 = y2[(timeperiod - 1):]
        y = y2

        print('y2',y2)

    #print('yn',yn1[10:20],yn2[10:20],yn3[10:20], 'y2\n',y2[10:20])

    #print('shapes',X.shape, y2.shape, y.shape)

    #print('y',data['close'][1:].reset_index(drop=True), data['close'][:-1].reset_index(drop=True))

    assert (X.shape[0] == y.shape[0])
    return X, y


if __name__ == '__main__':

    y_acc = []
    x_ind = []
    f = open('Tree_map.txt', 'w')

    for i in range(8):
        # './rev_Binance_BTCUSDT_d.csv': 1300 days worth data (time unit = day)
        indicator_sets_ref = {1: 'sma21', 2: 'sma50', 3: 'ema21', 4: 'ema50', 5: 'rsi14', 6: 'mfi10', 7:'adx', 8:'atr'}
        # Select 3 indices from the above list of indicators for running the algorithm

        f.writelines("1: 'sma21', 2: 'sma50', 3: 'ema21', 4: 'ema50', 5: 'rsi14', 6: 'mfi10', 7:'adx', 8:'atr'")
        f.writelines('\n')
        #indicator_list = [1,2,3]
        keys = list(indicator_sets_ref.keys())
        #print('dd',list(np.random.choice(keys,3,replace=True)))
        indicator_list = list(np.random.choice(keys,3,replace=False))
        X, y = gen_X_y(csv_path='./csvs/rev_Binance_BTCUSDT_d.csv', timeperiod=60, indicator_list=indicator_list, long=True)
        accuracy = 0
        # Split train/test sets
        iters = 1
        for c in range(iters):
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            print('train,test',X_train.shape, X_test.shape, y_train.shape, y_test.shape)

            dt_clf = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=25).fit(X_train, y_train)

            # Predict class labels
            y_test_pred = dt_clf.predict(X_test)
            # Predict probability for each class label (Could be used as `confidence` measure)
            #y_conf = dt_clf.predict_proba(X_test)
            #print('y conf',y_conf)

            plt.figure(0)
            tree.plot_tree(dt_clf)
            plt.show()

            from sklearn.tree import export_text
            r = export_text(dt_clf, feature_names=indicator_list)

            f.writelines(r)
            f.writelines('\n')

            print('r',r)

            accuracy = accuracy + accuracy_score(y_test, y_test_pred)

        accuracy_mean = accuracy/iters
        y_acc.append(accuracy_mean*100)
        x_ind.append(str(indicator_list[0])+'-'+str(indicator_list[1])+'-'+str(indicator_list[2]))

        acc_str = 'classification accuracy for {0}, {1}, {2} :{3} '.format(indicator_sets_ref[indicator_list[0]],
                                                             indicator_sets_ref[indicator_list[1]],
                                                             indicator_sets_ref[indicator_list[2]],
                                                             accuracy_mean)
        f.writelines(acc_str)
        f.writelines('\n')

        print('classification accuracy for {0}, {1}, {2} :{3} '.format(indicator_sets_ref[indicator_list[0]],
                                                             indicator_sets_ref[indicator_list[1]],
                                                             indicator_sets_ref[indicator_list[2]],
                                                             accuracy_mean))

    print('x and y',x_ind,y_acc)
    f.writelines('Indicator sets:',x_ind)
    f.writelines('\n')
    f.writelines('Accuracy :', y_acc)
    f.writelines('\n')
    #plt.xlim(0,10)
    plt.ylim(40,100)
    plt.xlabel('Indicator set')
    plt.ylabel('mean accuracy')
    plt.plot(x_ind,y_acc,label="1:sma21, 2:sma50, 3:ema21, 4:ema50, \n 5: rsi14, 6: mfi10, 7:adx14, 8:atr14")
    plt.legend(loc="upper right")
    plt.show()
    f.close()