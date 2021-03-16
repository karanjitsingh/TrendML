import backtrader as bt
import datetime
import matplotlib

class RSIStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.talib.RSI(self.data, period = 30)
    
    def next(self):
        if self.rsi < 30 and not self.position:
            self.buy(size = 1)
        if self.rsi > 70 and self.position:
            self.close()

def init_cerebro_with_data(datapath):
    cerebro = bt.Cerebro()
    # docs on how to set parameters properly to parse .csv files:
    # https://www.backtrader.com/docu/datafeed/
    data = bt.feeds.GenericCSVData(
        dataname=datapath,
        fromdate = datetime.datetime(2017, 8, 18),
        todate = datetime.datetime(2021, 3, 6),
        # Following args are column indices
        datetime = 1,
        time = -1,
        open = 4,
        high = 5,
        low = 6,
        close = 7,
        volume = 8,
        dtformat = "%Y-%m-%d"
    )
    cerebro.adddata(data)
    return cerebro

def run_backtest(datapath):
    cerebro = init_cerebro_with_data(datapath)
    cerebro.addstrategy(RSIStrategy)
    
    print('Starting portfolio value: $%.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final portfolio value: $%.2f' % cerebro.broker.getvalue())
    cerebro.plot()

if __name__ == '__main__':
    run_backtest('../decision_tree/csvs/rev_Binance_BTCUSDT_d.csv')