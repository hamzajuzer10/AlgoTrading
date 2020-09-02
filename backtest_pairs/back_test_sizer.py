import backtrader as bt
from datetime import datetime
import math


class exampleSizer(bt.Sizer):
    params = (('size',1),)
    def _getsizing(self, comminfo, cash, data, isbuy):
        return self.p.size

class printSizingParams(bt.Sizer):
    '''
    Prints the sizing parameters and values returned from class methods.
    '''
    def _getsizing(self, comminfo, cash, data, isbuy):
        #Strategy Method example
        pos = self.strategy.getposition(data)
        #Broker Methods example
        acc_value = self.broker.getvalue()

        #Print results
        print('----------- SIZING INFO START -----------')
        print('--- Strategy method example')
        print(pos)
        print('--- Broker method example')
        print('Account Value: {}'.format(acc_value))
        print('--- Param Values')
        print('Cash: {}'.format(cash))
        print('isbuy??: {}'.format(isbuy))
        print('data[0]: {}'.format(data[0]))
        print('------------ SIZING INFO END------------')

        return 0

class maxRiskSizer(bt.Sizer):
    '''
    Returns the number of shares rounded down that can be purchased for the
    max rish tolerance
    '''
    params = (('risk', 0.5),)

    def __init__(self):
        if self.p.risk > 1 or self.p.risk < 0:
            raise ValueError('The risk parameter is a percentage which must be'
                'entered as a float. e.g. 0.5')

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy == True:
            size = math.floor((cash * self.p.risk) / data[0])
        else:
            size = math.floor((cash * self.p.risk) / data[0]) * -1
        return size