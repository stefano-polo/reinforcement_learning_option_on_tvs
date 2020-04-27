import numpy as np
import random as rnd
from scipy.interpolate import interp1d
import scipy.stats as si  #for gaussian cdf
from scipy import exp, log, sqrt


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
        martingale = np.ones((Nsim,len(fixings)))
        if fixings[0] != 0.0:
            Z = np.random.normal(0,1,int(Nsim*0.5))
            Z = np.concatenate((Z,-Z)) #antithetic sampling
            martingale.T[0] = exp(-0.5*(self.volatility**2)*(fixings[0])+self.volatility*sqrt(fixings[0])*Z)
        for i in range (1,len(fixings)):
                Z = np.random.normal(0,1,int(Nsim*0.5))
                Z = np.concatenate((Z,-Z))   
                martingale.T[i] = martingale.T[i-1]*exp(-0.5*(self.volatility**2)*(fixings[i]-fixings[i-1])+self.volatility*sqrt(fixings[i]-fixings[i-1])*Z)

        return martingale*self.forward_curve(fixings)

    def Vanilla_PayOff(self,St=None,strike=None, typo = 1): #Monte Carlo call payoff
        zero = np.zeros((len(St),len(St.T)))
        if typo ==1:
            """Call option"""
            pay = St-strike
        elif typo ==-1:
            """Put option"""
            pay = strike - St
        pay1,pay2 = np.split(np.maximum(pay,zero),2) #for antithetic sampling
        return 0.5*(pay1+pay2)
    
    def Asian_PayOff(self,GM=None,strike=None, typo = 1): #Monte Carlo call payoff
        zero = np.zeros(len(GM))
        if typo ==1:
            """Call option"""
            pay = GM-strike
        elif typo ==-1:
            """Put option"""
            pay = strike - GM
        pay1, pay2 = np.split(np.maximum(pay,zero),2)
        return 0.5*(pay1+pay2)