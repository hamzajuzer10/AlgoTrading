from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import backtrader as bt
from datetime import datetime

ticker = 'KFYP'
start_date = '2020-12-01'
end_date = '2020-12-31'

class St(bt.Strategy):
    def logdata(self):
        txt = []
        txt.append('{}'.format(len(self)))

        txt.append('{}'.format(
            self.data.datetime.datetime(0).isoformat())
        )
        txt.append('{:.2f}'.format(self.data.open[0]))
        txt.append('{:.2f}'.format(self.data.high[0]))
        txt.append('{:.2f}'.format(self.data.low[0]))
        txt.append('{:.2f}'.format(self.data.close[0]))
        txt.append('{:.2f}'.format(self.data.volume[0]))
        print(','.join(txt))

    def next(self):
        self.logdata()


def run(ticker: str, from_date: str, to_date: str, save_file: str = None, save_folder: str = None):

    """Get historical data from IB - need to login to paper trading account in order for connection to establish"""

    from_date = datetime.strptime(from_date, "%Y-%m-%d").date()
    to_date = datetime.strptime(to_date, "%Y-%m-%d").date()

    cerebro = bt.Cerebro(stdstats=False)
    store = bt.stores.IBStore(port=7497)

    stockkwargs = dict(
        timeframe=bt.TimeFrame.Days, # use daily timeframe
        rtbar=False,  # use RealTime 5 seconds bars
        historical=True,  # only historical download
        #what='TRADES',
        qcheck=0.5,  # timeout in seconds (float) to check for events
        fromdate=from_date, # datetime(2019, 9, 24),  # get data from.. (non-inclusive)
        todate=to_date, # datetime(2019, 9, 26),  # get data from..
        latethrough=True,  # let late samples through
        backfill_start=True,
        tradename=None  # use a different asset as order target
    )

    data0 = store.getdata(dataname=ticker, **stockkwargs)

    cerebro.resampledata(data0, timeframe=bt.TimeFrame.Days, compression=1)

    cerebro.addstrategy(St)
    cerebro.run()

if __name__ == '__main__':

    ## TODO: Issue - data downloads dont match Trader Workstation close prices and Yahoo - need to sort

    run(ticker=ticker,
        from_date=start_date,
        to_date=end_date)