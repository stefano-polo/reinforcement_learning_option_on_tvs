import numpy as np
from scipy.interpolate import interp1d
from scipy import exp, sqrt, log, heaviside
from scipy.integrate import quad
from numpy.linalg import cholesky

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
        self.T = repo_dates
        self.q = np.array([repo_rates[0]]) 
        for i in range(1,len(self.T)):
            alpha = ((self.T[i]-self.reference)*(repo_rates[i])-(self.T[i-1]-self.reference)*
                     (repo_rates[i-1]))/(self.T[i]-self.T[i-1])
            self.q = np.append(self.q,alpha)
        print("Forward repo time grid",self.T)
        print("Forward repo rate: ", self.q)

    def curve(self, date):
        q = piecewise_function
        date = np.array(date)
        if date.shape!=():
            return  np.asarray([(self.spot/self.discounting_curve(extreme))*exp(-quad(q,self.reference,extreme,args=(self.T,self.q),limit=100)[0]) for extreme in date])
        else:
            return (self.spot/self.discounting_curve(date))*exp(-quad(q,self.reference,date,args=(self.T,self.q),limit=100)[0])


class DiscountingCurve(Curve):

    def __init__(self, reference=None, discounts=None, dates=None):
        self.reference = reference
        self.T = dates
        r_zero = (1./(self.reference-dates))*log(discounts)
        self.r = np.array([r_zero[0]]) #zero rate from 0 to T1
        for i in range(1,len(self.T)):
            alpha = ((self.T[i]-self.reference)*(r_zero[i])-(self.T[i-1]-self.reference)*
                     (r_zero[i-1]))/(self.T[i]-self.T[i-1])
            self.r = np.append(self.r,alpha)
        print("Forward interest rate time grid",self.T)
        print("Forward interest rate: ", self.r)

    def curve(self, date):
        r = piecewise_function
        date = np.array(date)
        if date.shape!=():
            return  np.asarray([exp(-quad(r,self.reference,extreme,args=(self.T,self.r),limit=100)[0]) for extreme in date])
        else:
            return exp(-quad(r,self.reference,date,args=(self.T,self.r),limit=100)[0])
  

class ForwardVariance(Curve):  #I calculate the variance and not the volatility for convenience of computation

    def __init__(self, reference=None, spot_volatility=None, strikes=None, maturities=None, strike_interp=None):
        self.reference = reference #pricing date of the implied volatilities
        self.T = maturities
        self.spot_vol = interp1d(strikes,spot_volatility,axis=0)(strike_interp)
        self.forward_vol = np.array([self.spot_vol[0]]) #forward volatility from 0 to T1
        for i in range (1,len(self.T)):
            alpha = ((self.T[i]-self.reference)*(self.spot_vol[i]**2)-(self.T[i-1]-self.reference)*
                     (self.spot_vol[i-1]**2))/(self.T[i]-self.T[i-1])
            self.forward_vol = np.append(self.forward_vol, sqrt(alpha))
        print("Forward volatility time grid: ",self.T)
        print("Forward volatility: ",self.forward_vol)

    def curve(self,date):
        return piecewise_function(date,self.T,self.forward_vol**2)


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

    def simulate(self, fixings=None, Ndim = 1, corr = None, Nsim=1, seed=14,**kwargs):
        np.random.seed(seed)
        Nsim = int(Nsim)
        if Ndim == 1:
            print("Single Asset Simulation")
            logmartingale = np.zeros((int(2*Nsim),len(fixings)))
            for i in range (len(fixings)):
                Z = np.random.normal(0,1,Nsim)
                Z = np.concatenate((Z,-Z))
                if i ==0:
                    logmartingale.T[i]=-0.5*quad(self.variance,0,fixings[i])[0]+sqrt(quad(self.variance,0,fixings[i])[0])*Z
                elif i!=0:
                    logmartingale.T[i]=logmartingale.T[i-1]-0.5*quad(self.variance,fixings[i-1],fixings[i])[0]+sqrt(quad(self.variance,fixings[i-1],fixings[i])[0])*Z
            return exp(logmartingale)*self.forward_curve(fixings)

        else:
            print("Multi Asset Simulation")
            logmartingale = np.zeros((int(2*Nsim),len(fixings),Ndim))
            R = cholesky(corr)
            for i in range (len(fixings)):
                Z = np.random.normal(0,1,(Nsim,Ndim))
                Z = np.concatenate((Z,-Z)) #matrix of uncorrelated random variables
                ep = np.dot(R,Z.T)   #matrix of correlated random variables
                for j in range(Ndim):
                    if i ==0:
                        logmartingale[:,i,j]=-0.5*quad(self.variance[j],0,fixings[i])[0]+sqrt(quad(self.variance[j],0,fixings[i])[0])*ep[j]
                    elif i!=0:
                        logmartingale[:,i,j]=logmartingale[:,i-1,j]-0.5*quad(self.variance[j],fixings[i-1],fixings[i])[0]+sqrt(quad(self.variance[j],fixings[i-1],fixings[i])[0])*ep[j]
            M = exp(logmartingale)
            for i in range(Ndim):
                M[:,:,i] = M[:,:,i]*self.forward_curve[i](fixings)
            return M


"""Payoff Functions"""
def Vanilla_PayOff(St=None,strike=None, typo = 1): #Monte Carlo call payoff
    zero = np.zeros(St.shape)
    if typo ==1:
        """Call option payoff"""
        pay = St-strike
    elif typo ==-1:
        """Put option payoff"""
        pay = strike - St
    pay1,pay2 = np.split(np.maximum(pay,zero),2) #for antithetic sampling
    return 0.5*(pay1+pay2)


"""Definition of a piecewise function"""
def piecewise_function(date,interval,value):
    if value.ndim == 3:   #matrix piecewise function
        date = np.array(date)
        if date.shape!=():  #vector input
            val = np.ones(len(date))
            y = (heaviside(date,val)-heaviside(date-interval[0],val))*(value[:,:,0])
            for i in range(1,len(interval)):
                y = y + (heaviside(date-interval[i-1],val)-heaviside(date-interval[i],val))*(value[:,:,i])
            return y
        else:
            val = 1  #scalar input
            y = (heaviside(date,val)-heaviside(date-interval[0],val))*(value[:,:,0])
            for i in range(1,len(interval)):
                y = y + (heaviside(date-interval[i-1],val)-heaviside(date-interval[i],val))*(value[:,:,i])
            return y
    else:
        date = np.array(date)
        if date.shape!=():  #vector input
            val = np.ones(len(date))
            y = (heaviside(date,val)-heaviside(date-interval[0],val))*(value[0])
            for i in range(1,len(interval)):
                y = y + (heaviside(date-interval[i-1],val)-heaviside(date-interval[i],val))*(value[i])
            return y
        else:
            val = 1  #scalar input
            y = (heaviside(date,val)-heaviside(date-interval[0],val))*(value[0])
            for i in range(1,len(interval)):
                y = y + (heaviside(date-interval[i-1],val)-heaviside(date-interval[i],val))*(value[i])
            return y
