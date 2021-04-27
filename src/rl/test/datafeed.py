from gym.spaces import Discrete

from tensortrade.env.default.actions import TensorTradeActionScheme

from tensortrade.env.generic import ActionScheme, TradingEnv
from tensortrade.core import Clock
from tensortrade.oms.instruments import ExchangePair
from tensortrade.oms.wallets import Portfolio
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.orders import (
    Order,
    proportion_order,
    TradeSide,
    TradeType
)
from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.agents import DQNAgent

import tensortrade.env.default as default

from ray import tune
from ray.tune.registry import register_env

# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
# def get_available_cpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'CPU']
# num_gpus = len(get_available_gpus())
# num_cpus = len(get_available_cpus())


from tensortrade.oms.instruments import USD, BTC, ETH

def getPortfolio(exchange, usd=50000, btc=1):
    return Portfolio(USD, [
        Wallet(exchange, usd * USD),
        Wallet(exchange, btc * BTC)
    ])


# Fetch Data
cdd = CryptoDataDownload()
data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")

def CPLogRSIMacdFeed():

    
    def rsi(price: Stream[float], period: float) -> Stream[float]:
        r = price.diff()
        upside = r.clamp_min(0).abs()
        downside = r.clamp_max(0).abs()
        rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
        return 100*(1 - (1 + rs) ** -1)


    def macd(price: Stream[float], fast: float, slow: float, signal: float) -> Stream[float]:
        fm = price.ewm(span=fast, adjust=False).mean()
        sm = price.ewm(span=slow, adjust=False).mean()
        md = fm - sm
        signal = md - md.ewm(span=signal, adjust=False).mean()
        return signal


    features = []
    for c in data.columns[1:]:
        s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
        features += [s]

    cp = Stream.select(features, lambda s: s.name == "close")

    features = [
        # Remove auto correlation
        cp.log().diff().rename("lr"),
        rsi(cp, period=20).rename("rsi"),
        macd(cp, fast=10, slow=50, signal=5).rename("macd")
    ]

    feed = DataFeed(features)
    feed.compile()


    return feed

def create_env(config):
    # p = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")

    # # Define exchange
    # bitfinex = Exchange("bitfinex", service=execute_order)(
    #     p
    # )

    # portfolio = getPortfolio(bitfinex, 50000, 1)

    feed = CPLogRSIMacdFeed()

    for i in range(5):
        print(feed.next())

    reward_scheme = PBR(price=p)

    action_scheme = BuySellHold(
        cash=50000,
        asset=1
    ).attach(reward_scheme)

    renderer_feed = DataFeed([
        Stream.source(list(data["date"])).rename("date"),
        Stream.source(list(data["open"]), dtype="float").rename("open"),
        Stream.source(list(data["high"]), dtype="float").rename("high"),
        Stream.source(list(data["low"]), dtype="float").rename("low"),
        Stream.source(list(data["close"]), dtype="float").rename("close"),
        Stream.source(list(data["volume"]), dtype="float").rename("volume")
    ])

    environment = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        renderer_feed=renderer_feed,
        renderer=default.renderers.PlotlyTradingChart(),
        window_size=config["window_size"],
        max_allowed_loss=0.6
    )
    return environment

register_env("TradingEnv", create_env)
config = {
    'window_size': 25
}
create_env(config)