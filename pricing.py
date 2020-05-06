import numpy as np
from scipy.interpolate import interp1d
from scipy import exp, sqrt, log, heaviside
from scipy.integrate import quad


"""Classes for my simulation"""

class Curve:

    def __init__(self, **kwargs):

        raise Exception('do not instantiate this class.')

    def __call__(self, date): #return the value of the curve at a defined time

        return self.curve(date)


class EquityForwardCurve(Curve):

    def __init__(self, spot=None, reference=None, discounting_curve=None,
                repo_rates=None, repo_dates=None):
        self.spot = spot
        self.reference = reference
        self.discounting_curve = discounting_curve
        self.q = interp1d(repo_dates,repo_rates)   #linear interpolation for repo repo_rates

    def curve(self, date):
        return (self.spot/self.discounting_curve(date))*exp(-self.q(date-self.reference)*(date-self.reference))


class DiscountingCurve(Curve):

    def __init__(self, reference=None, discounts=None, dates=None):

        self.reference = reference
        self.r = interp1d(dates,(-1./dates)*log(discounts)) #linear interpolation of the zero_rates

    def curve(self, date):
        return exp(-self.r(date-self.reference)*(date-self.reference))


class ForwardVariance(Curve):  #I calculate the variance and not the volatility for convenience of computation in the GBM

    def __init__(self, reference=None, spot_volatility=None, strikes=None, maturities=None, strike_interp=None):
        self.reference = reference #pricing date of the implied volatilities
        self.T = maturities
        self.spot_vol = interp1d(strikes,spot_volatility,axis=0)(strike_interp)
        self.forward_vol = np.array([self.spot_vol[0]]) #forward volatility from 0 to T1
        for i in range (1,len(self.T)):
            alpha = ((self.T[i]-self.reference)*(self.spot_vol[i]**2)-(self.T[i-1]-self.reference)*
                     (self.spot_vol[i-1]**2))/(self.T[i]-self.T[i-1])
            self.forward_vol = np.append(self.forward_vol, sqrt(alpha))
        print(self.forward_vol)

    def curve(self,date):
        date = np.array(date)
        if date.shape!=():  #date = vector input
            val = np.ones(len(date))
            y = (heaviside(date,val)-heaviside(date-self.T[0],val))*(self.forward_vol[0]**2)
            for i in range(1,len(self.T)):
                y = y + (heaviside(date-self.T[i-1],val)-heaviside(date-self.T[i],val))*(self.forward_vol[i]**2)
            return y
        else:
            val = 1  #date = scalar input
            y = (heaviside(date,val)-heaviside(date-self.T[0],val))*(self.forward_vol[0]**2)
            for i in range(1,len(self.T)):
                y = y + (heaviside(date-self.T[i-1],val)-heaviside(date-self.T[i],val))*(self.forward_vol[i]**2)
            return y


class PricingModel:

    def __init__(self, **kwargs):

      raise Exception('model not implemented.')

    def simulate(self, fixings=None, Nsim=1, seed=14, **kwargs):

      raise Exception('simulate not implemented.')


class Black(PricingModel):
    """Black model"""

    def __init__(self, variance=None, forward_curve=None,**kwargs):
        self.variance = variance
        self.forward_curve = forward_curve

    def simulate(self, fixings=None, Nsim=1, seed=14,**kwargs):
        np.random.seed(seed)
        Nsim = int(Nsim)
        logmartingale = np.zeros((Nsim,len(fixings)))
        for i in range (len(fixings)):
            Z = np.random.normal(0,1,int(Nsim*0.5))
            Z = np.concatenate((Z,-Z))
            if i ==0:
                logmartingale.T[i]=-0.5*quad(self.variance,0,fixings[i])[0]+sqrt(quad(self.variance,0,fixings[i])[0])*Z
            elif i!=0:
                logmartingale.T[i]=logmartingale.T[i-1]-0.5*quad(self.variance,fixings[i-1],fixings[i])[0]+sqrt(quad(self.variance,fixings[i-1],fixings[i])[0])*Z

        return exp(logmartingale)*self.forward_curve(fixings)

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
