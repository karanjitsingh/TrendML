
import backtrader as bt
import datetime
import matplotlib


class RSIStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.talib.RSI(self.data, period=14)

    def next(self):
        if self.rsi >= 60 and not self.position:
            self.buy(size=1)
        if self.rsi < 60 and self.position:
            self.close()


class SMAStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.talib.SMA(self.data, period=21)

    def next(self):
        if self.sma < self.data and not self.position:
            self.buy(size=1)
        if self.sma > self.data and self.position:
            self.close()


class SMA2Strategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.talib.SMA(self.data, period=50)

    def next(self):
        if self.sma < self.data and not self.position:
            self.buy(size=1)
        if self.sma > self.data and self.position:
            self.close()


class EMAStrategy(bt.Strategy):
    def __init__(self):
        self.ema = bt.talib.EMA(self.data, period=21)

    def next(self):
        if self.ema < self.data and not self.position:
            self.buy(size=1)
        if self.ema > self.data and self.position:
            self.close()


class EMA2Strategy(bt.Strategy):
    def __init__(self):
        self.ema = bt.talib.EMA(self.data, period=50)

    def next(self):
        if self.ema < self.data and not self.position:
            self.buy(size=1)
        if self.ema > self.data and self.position:
            self.close()


class adxStrategy(bt.Strategy):
    def __init__(self):

        self.adx = bt.talib.ADX(self.data.high, self.data.low, self.data.close, period=14)
    def next(self):
        if self.adx >35 and not self.position:
            self.buy(size=1)
        if self.adx <35 and self.position:
            self.close()


class atrStrategy(bt.Strategy):
    def __init__(self):

        self.atr = bt.talib.ADX(self.data.high, self.data.low, self.data.close, period=14)
    def next(self):
        if self.atr > (self.data.high - self.data.low) and not self.position:
            self.buy(size=1)
        if self.atr < (self.data.high - self.data.low) and self.position:
            self.close()


class mfiStrategy(bt.Strategy):
    def __init__(self):

        self.mfi = bt.talib.ADX(self.data.high, self.data.low, self.data.close, period=14)
    def next(self):
        if self.mfi >25 and not self.position:
            self.buy(size=1)
        if self.mfi <25 and self.position:
            self.close()



class Turtle_trend(bt.Strategy):
    params = (('period1', 21), ('period2', 10))
    def __init__(self):
        self.ema = bt.talib.EMA(self.data, period=21)
        self.init_mark = bt.Max(self.data.get(size=self.params.period1))
        self.close_mark = bt.Max(self.data.get(size=self.params.period2))
        print('types',self.init_mark)

    def next(self):
        if self.data > self.init_mark and not self.position:
            self.buy(size=1)
        if self.data < self.close_mark and self.position:
            self.close()



def init_cerebro_with_data(datapath):
    cerebro = bt.Cerebro()
    # docs on how to set parameters properly to parse .csv files:
    # https://www.backtrader.com/docu/datafeed/
    data = bt.feeds.GenericCSVData(
        dataname=datapath,
        fromdate=datetime.datetime(2017, 8, 18),
        todate=datetime.datetime(2020, 3, 6),
        # Following args are column indices
        datetime=1,
        time=-1,
        open=4,
        high=5,
        low=6,
        close=7,
        volume=8,
        dtformat="%d-%m-%Y"
    )
    cerebro.adddata(data)
    return cerebro


def run_backtest(datapath):
    cerebro = init_cerebro_with_data(datapath)
    #cerebro.addstrategy(SMAStrategy)
    cerebro.addstrategy(SMA2Strategy)
    #cerebro.addstrategy(EMAStrategy)
    #cerebro.addstrategy(EMA2Strategy)
    #cerebro.addstrategy(RSIStrategy)
    #cerebro.addstrategy(adxStrategy)
    cerebro.addstrategy(atrStrategy)
    #cerebro.addstrategy(mfiStrategy)
    #cerebro.addstrategy(Turtle_trend)

    print('Starting portfolio value: $%.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final portfolio value: $%.2f' % cerebro.broker.getvalue())
    #cerebro.plot()


if __name__ == '__main__':
    run_backtest('../decision_tree/csvs/rev_Binance_BTCUSDT_d.csv')