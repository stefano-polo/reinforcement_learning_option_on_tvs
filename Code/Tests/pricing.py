import numpy as np
import random as rnd
from scipy.interpolate import interp1d
import scipy.stats as si  #for gaussian cdf
from scipy import exp, log, sqrt


"""Functions for Black & Scholes Formula"""
def d1(S, K, T, r, q, sigma):
    """d_1 function for BS model"""
    return (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

def d2(S, K, T, r, q, sigma):
    """d_2 function for BS model"""
    return  (log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

def Call_Closed(S, K, T, r, q, sigma):
    """Closed form of Call Option for BS model"""
    d_1 = d1(S, K, T, r, q, sigma)
    d_2 = d2(S, K, T, r, q, sigma)
    return S * exp(-q * T) * si.norm.cdf(d_1, 0.0, 1.0) - K * exp(-r * T) * si.norm.cdf(d_2, 0.0, 1.0)

def vega(S, K, T, r, q, sigma):
    """Vega"""
    d_1 = d1(S, K, T, r, q, sigma)
    return (1 / sqrt(2 * np.pi)) * S * exp(-q * T) * sqrt(T) * exp((-d_1 ** 2) * 0.5)


"""Classes for my simulation"""

class Curve:

    def __init__(self, **kwargs):

        raise Exception('do not instantiate this class.')

    def __call__(self, date): #return the value of the curve at a defined time

        return self.curve(date)


class EquityForwardCurve(Curve):

    def __init__(self, spot=None, reference=None, discounting_curve=None,
                repo_rates=None, dividend_yelds=None): #discounting_curve is a a DiscountingCurve type object
        self.spot = spot
        self.reference = reference  #pricing date (t_0)
        self.discounting_curve = discounting_curve
        self.q = repo_rates
        self.d = dividend_yelds


    def curve(self, date):
        return (self.spot/self.discounting_curve(date))*exp((self.q-self.d)*(date-self.reference))




class DiscountingCurve(Curve):

    def __init__(self, reference=None, zero_rate = None):

        self.reference = reference
        self.r = zero_rate

    def curve(self, date):
        return exp(-self.r*(date-self.reference))


class PricingModel:

    def __init__(self, **kwargs):

      raise Exception('model not implemented.')

    def simulate(self, fixings=None, Nsim=1, seed=14, **kwargs):

      raise Exception('simulate not implemented.')


class Black(PricingModel):
    """Black model"""

    def __init__(self, volatility=None, forward_curve=None,**kwargs):
        self.volatility = volatility
        self.forward_curve = forward_curve

    def simulate(self, fixings=None, Nsim=1, seed=14,**kwargs):
        np.random.seed(seed)
        Nsim = int(Nsim)
        martingale = np.zeros((Nsim,len(fixings)))
        for i in range (len(fixings)):
                Z = np.random.normal(0,1,int(Nsim*0.5))
                Z = np.concatenate((Z,-Z))   #antithetic sampling
                martingale.T[i] = exp(-0.5*(self.volatility**2)*fixings[i]+self.volatility*sqrt(fixings[i])*Z)

        return martingale*self.forward_curve(fixings)

    def Call_PayOff(self,St,strike): #Monte Carlo call payoff
        zero = np.zeros((len(St),len(St.T)))
        pay = St-strike
        pay1,pay2 = np.split(np.maximum(pay,zero),2) #for antithetic sampling
        return 0.5*(pay1+pay2)
