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
        self.T = ACT_360(repo_dates,self.reference)
        self.q = np.array([repo_rates[0]]) 
        for i in range(1,len(self.T)):
            alpha = ((self.T[i])*(repo_rates[i])-(self.T[i-1])*
                     (repo_rates[i-1]))/(self.T[i]-self.T[i-1])
            self.q = np.append(self.q,alpha)
        print("Forward repo time grid",self.T)
        print("Forward repo rate: ", self.q)

    def curve(self, date):
        q = piecewise_function
        date = np.array(date)
        if date.shape!=():
            return  np.asarray([(self.spot/self.discounting_curve(extreme))*exp(-quad(q,0,extreme,args=(self.T,self.q),limit=200)[0]) for extreme in date])
        else:
            return (self.spot/self.discounting_curve(date))*exp(-quad(q,0,date,args=(self.T,self.q),limit=200)[0])    
    
    
class DiscountingCurve(Curve):

    def __init__(self, reference=None, discounts=None, dates=None):
        self.reference = reference
        self.T = ACT_365(dates,self.reference)
        if self.T[0] ==0:
            r_zero = np.array([0])   #at reference date the discount is 1
            r_zero = np.append(r_zero,(1./((self.reference-dates[1:])/365))*log(discounts[1:]))
        else:
            r_zero = (1./((self.reference-dates)/365))*log(discounts)
        self.r = interp1d(self.T,r_zero) #zero rate from 0 to T1
        print("zero interest rate time grid",self.T)
        print("zero interest rate: ",r_zero)

    def curve(self, date):
        return exp(-self.r(date)*date)
  

class ForwardVariance(Curve):  #I calculate the variance and not the volatility for convenience of computation

    def __init__(self, reference=None, spot_volatility=None, strikes=None, maturities=None, forward=None):
        self.reference = reference #pricing date of the implied volatilities
        self.T = ACT_365(maturities,self.reference)
        matrix_interpolated = interp1d(strikes,spot_volatility,axis=1)(forward(self.T))
        self.spot_vol = np.array([])
        for i in range (len(maturities)):
            self.spot_vol = np.append(self.spot_vol,matrix_interpolated[i,i])
        self.forward_vol = np.array([self.spot_vol[0]]) #forward volatility from 0 to T1
        for i in range (1,len(self.T)):
            alpha = ((self.T[i])*(self.spot_vol[i]**2)-(self.T[i-1])*
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

def ACT_365(date1,date2):
    """Day count convention for a normal year"""
    return abs(date1-date2)/365

def ACT_360(date1,date2):
    """Day count convention for a 360 year"""
    return abs(date1-date2)/360