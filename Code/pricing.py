import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from scipy.interpolate import interp1d
import scipy.stats as si  #for gaussian cdf


def d1(S, K, T, r, q, sigma):
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def Call_Closed(S, K, T, r, q, d_1, d_2):
    return S * np.exp(-q * T) * si.norm.cdf(d_1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d_2, 0.0, 1.0)

def vega(S,T, q, d_1):
    return (1 / np.sqrt(2 * np.pi)) * S * np.exp(-q * T) * np.sqrt(T) * np.exp((-d_1 ** 2) * 0.5)

class Curve:

    def __init__(self, **kwargs):

        raise Exception('do not instantiate this class.')

    def __call__(self, date): #return the value of the curve at a defined time

        return self.curve(date)


class DiscountingCurve(Curve):
    def __init__(self, reference=None, discounts=None, dates=None):
        self.reference = reference
        self.discounts = interp1d(dates, discounts)

    def curve(self, date):
        return np.exp(-self.discounts(date)*date)


class EquityForwardCurve(Curve):

    def __init__(self, spot=None, reference=None, discounting_curve=None,
                repo_rates=None, repo_dates=None, dividend_rates=None, dividend_dates=None): #discounting_curve is a a DiscountingCurve type object
        self.spot = spot
        self.reference = reference
        self.discounting_curve = discounting_curve
        #self.repo_rates = interp1d(repo_dates, repo_rates)  #Linear interpolation inside the constructor since it is done only once
        #self.dividend_rates = interp1d(dividend_dates, dividend_rates)

    def curve(self, date):
        return (self.spot/self.discounting_curve(date))#*np.exp(-(self.dividend_rates(date)+self.repo_rates(date))*date)




class PricingModel:
    def __init__(self, **kwargs):
        raise Exception('model not implemented.')

    def simulate(self, fixings=None, Nsim=1, seed=14, **kwargs):
        raise Exception('simulate not implemented.')


class Black(PricingModel):
    """Black model"""

    def __init__(self, volatility=None, forward_curve=None, fixings=None, **kwargs):
        self.volatility = volatility
        self.forward_curve = forward_curve(fixings)  #K/F_0(T)
        self.fixings = fixings

    def setSeed(self, seed):
        rnd.seed(seed)

    def simulate(self, Nsim,**kwargs): #simulation of Martingale
        self.martingale = np.zeros((Nsim,len(self.fixings)))
        for i in range (Nsim):
            self.martingale[i] = np.exp(-0.5*(self.volatility**2)*self.fixings[:]+self.volatility*np.sqrt(self.fixings[:])*rnd.gauss(0,1))

    def Call_PayOff(self,strike): #Monte Carlo call
        self.h = strike/self.forward_curve
        self.pay = self.martingale- self.h
        for i in range (len(self.fixings)):
            for j in range (len(self.martingale)):
                self.pay[j][i] = max(self.pay[j][i],0)


    def newton_vol_call_div(self, S, C, i, r=0, q=0):
        """Implied volatility"""
        xold = 0
        xnew = np.sqrt((2*np.pi)/self.fixings[i])*(C[i]/S[i])
        tolerance = 0.0001

        while abs(xnew - xold) > tolerance:
            xold = xnew
            d_1 = d1(S[i], self.h[i], self.fixings[i], r, q, xold)
            d_2 = d_1 - xold*np.sqrt(self.fixings[i])
            f = Call_Closed(S[i], self.h[i], self.fixings[i], r, q, d_1, d_2)
            v = vega(S[i], self.fixings[i], q, d_1)
            xnew = xold - (f-C[i])/ v

        return abs(xnew)
