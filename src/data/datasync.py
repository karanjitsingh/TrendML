import datastore
from datetime import datetime
synctime = datetime.fromtimestamp(1613775285.401)


# data.fetch("4h", synctime, 3 * 365 * 24 * 60, "BTCUSDT")
# data.fetch("6h", synctime, 3 * 365 * 24 * 60, "BTCUSDT")
# data.fetch("8h", synctime, 3 * 365 * 24 * 60, "BTCUSDT")
# data.fetch("12h", synctime, 3 * 365 * 24 * 60, "BTCUSDT")
# data.fetch("1d", synctime, 3 * 365 * 24 * 60, "BTCUSDT")
datastore.fetch("3d", synctime, 3 * 365 * 24 * 60, "BTCUSDT") 