import datastore
from datetime import datetime
synctime = datetime.fromtimestamp(1613775285.401)


# datastore.fetch("4h", synctime, 3 * 365 * 24 * 60, "BTCUSDT")
# datastore.fetch("6h", synctime, 3 * 365 * 24 * 60, "BTCUSDT")
# datastore.fetch("8h", synctime, 3 * 365 * 24 * 60, "BTCUSDT")
# datastore.fetch("12h", synctime, 3 * 365 * 24 * 60, "BTCUSDT")
datastore.fetch("1d", synctime, 3 * 365 * 24 * 60, "BTCUSDT")
# datastore.fetch("3d", synctime, 3 * 365 * 24 * 60, "BTCUSDT") 