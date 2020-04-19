import numpy as np
import random as rnd
from scipy.interpolate import interp1d
import scipy.stats as si  #for gaussian cdf


"""Functions for Black & Scholes Formula"""
def d1(S, K, T, r, q, sigma):
    """d_1 function for BS model"""
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, q, sigma):
    """d_2 function for BS model"""
    return  (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def Call_Closed(S, K, T, r, q, sigma):
    """Closed form of Call Option for BS model"""
    d_1 = d1(S, K, T, r, q, sigma)
    d_2 = d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * si.norm.cdf(d_1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d_2, 0.0, 1.0)

def vega(S, K, T, r, q, sigma):
    """Vega"""
    d_1 = d1(S, K, T, r, q, sigma)
    return (1 / np.sqrt(2 * np.pi)) * S * np.exp(-q * T) * np.sqrt(T) * np.exp((-d_1 ** 2) * 0.5)


"""Classes for my simulation"""

class Curve:

    def __init__(self, **kwargs):

        raise Exception('do not instantiate this class.')

    def __call__(self, date): #return the value of the curve at a defined time

        return self.curve(date)


class EquityForwardCurve(Curve):

    def __init__(self, spot=None, reference=None, discounting_curve=None,
                repo_rates=None, repo_dates=None, dividend_rates=None, dividend_dates=None): #discounting_curve is a a DiscountingCurve type object
        self.spot = spot
        self.reference = reference  #pricing date (t_0)
        self.discounting_curve = discounting_curve
        #self.repo_rates = interp1d(repo_dates, repo_rates)  #Linear interpolation inside the constructor since it is done only once
        #self.dividend_rates = interp1d(dividend_dates, dividend_rates)

    def curve(self, date):
        return (self.spot/self.discounting_curve(date))#*np.exp(-(self.dividend_rates(date)+self.repo_rates(date))*date)




class DiscountingCurve(Curve):

    def __init__(self, reference=None, discounts=None, dates=None):

        self.reference = reference
        self.discounts = interp1d(dates, discounts)

    def curve(self, date):
        return np.exp(-self.discounts(date-self.reference)*(date-self.reference))


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
        self.fixings = fixings# - forward_curve.reference  #self.fixings = t_i - t_0

    def setSeed(self, seed):
        rnd.seed(seed)

    def simulate(self, Nsim,**kwargs): #simulation of Martingale
        self.martingale = np.zeros((Nsim,len(self.fixings)))
        for i in range (Nsim):
            for j in range (len(self.fixings)):
                Z = rnd.gauss(0,1)
                self.martingale[i][j] = np.exp(-0.5*(self.volatility**2)*self.fixings[j]+self.volatility*np.sqrt(self.fixings[j])*Z)

    def Call_PayOff(self,strike): #Monte Carlo call payoff
        self.h = strike/self.forward_curve
        self.pay = self.martingale- self.h
        for i in range (len(self.martingale)):
            for j in range (len(self.fixings)):
                self.pay[i][j] = max(self.pay[i][j],0)

    def newton_implied_volatility(self, S, C, i, r=0, q=0):
        """Implied volatility"""
        xold = 0
        xnew = 0.23#np.sqrt((2*np.pi)/self.fixings[i])*(C[i]/S[i])
        tolerance = 0.0001
        counter = 0
        while abs(xnew - xold) > tolerance:
            xold = xnew
            xold = xnew
            f = Call_Closed(S[i], self.h[i], self.fixings[i], r, q, xold)
            v = vega(S[i], self.h[i], self.fixings[i], r, q, xold)
            xnew = xold - (f-C[i])/ v
            counter= counter +1
            if(counter>=1000):
                return None
        return abs(xnew)
