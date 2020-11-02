import numpy as np
from scipy.interpolate import interp1d
from numpy import exp, sqrt, log
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
                repo_rates=None, repo_dates=None, act = "No"):
        self.spot = spot
        self.reference = reference
        self.discounting_curve = discounting_curve
        if act =="360":
            self.T = ACT_360(repo_dates,self.reference)
            self.T = self.T*360/365
        elif act == "365":
            self.T = ACT_365(repo_dates,self.reference)
        else:
            self.T = abs(repo_dates - self.reference)
        self.q_values = np.array([repo_rates[0]])
        for i in range(1,len(self.T)):
            alpha = ((self.T[i])*(repo_rates[i])-(self.T[i-1])*
                     (repo_rates[i-1]))/(self.T[i]-self.T[i-1])
            self.q_values = np.append(self.q_values,alpha)
        if self.T[0] !=0:
            self.T = np.insert(self.T[:-1],0,0)
        self.q = interp1d(self.T, self.q_values, kind='previous',fill_value="extrapolate", assume_sorted=False) 
        print("Forward repo time grid",self.T)
        print("Forward repo rate: ", self.q_values)

    def curve(self, date):
        date = np.array(date)
        if date.shape!=():
            return  np.asarray([(self.spot/self.discounting_curve(extreme))*exp(-quad_piecewise(self.q,self.T,0,extreme)) for extreme in date])
        else:
            return (self.spot/self.discounting_curve(date))*exp(-quad_piecewise(self.q,self.T,0,date))


class DiscountingCurve(Curve):

    def __init__(self, reference=None, discounts=None, dates=None, act = "No"):
        self.reference = reference
        if act=="360":
            self.T = ACT_360(dates,self.reference)
        elif act == "365":
            self.T = ACT_365(dates,self.reference)
        else:
            self.T = abs(dates - self.reference)
        if self.T[0] ==0:
            r_zero = np.array([0])   #at reference date the discount is 1
            r_zero = np.append(r_zero,(-1./((self.T[1:])))*log(discounts[1:]))
            r_zero[0] = r_zero[1]
        else:
            r_zero = (-1./(self.T))*log(discounts)
        self.r = interp1d(self.T,r_zero) #zero rate from 0 to T1
        print("zero interest rate time grid",self.T)
        print("zero interest rate: ",r_zero)

    def curve(self, date):
        return exp(-self.r(date)*date)


class ForwardVariance(Curve):  #I calculate the variance and not the volatility for convenience of computation

    def __init__(self, reference=None, spot_volatility=None, strikes=None, maturities=None, strike_interp=None, act="No"):
        self.reference = reference #pricing date of the implied volatilities
        if act=="360":
            self.T = ACT_360(maturities,self.reference)
        elif act=="365":
            self.T = ACT_365(maturities,self.reference)
        else:
            self.T = abs(maturities-self.reference)
        if isinstance(strike_interp, EquityForwardCurve):
            """Interpolation with the ATM forward"""
            self.spot_vol = np.array([])
            matrix_interpolated = interp1d(strikes,spot_volatility,axis=1)(strike_interp(self.T))
            for i in range (len(maturities)):
                self.spot_vol = np.append(self.spot_vol,matrix_interpolated[i,i])
        else:
            """Interpolation with the ATM spot"""
            self.spot_vol = interp1d(strikes,spot_volatility,axis=0)(strike_interp)

        self.forward_vol = np.array([self.spot_vol[0]]) #forward volatility from 0 to T1
        for i in range (1,len(self.T)):
            alpha = ((self.T[i])*(self.spot_vol[i]**2)-(self.T[i-1])*
                     (self.spot_vol[i-1]**2))/(self.T[i]-self.T[i-1])
            self.forward_vol = np.append(self.forward_vol, sqrt(alpha))
        if self.T[0] !=0:
            self.T = np.insert(self.T[:-1],0,0)
        self.vol_t = interp1d(self.T, self.forward_vol, kind='previous',fill_value="extrapolate", assume_sorted=False) 
        print("Forward volatility time grid: ",self.T)
        print("Forward volatility: ",self.forward_vol)

    def curve(self,date):
        return self.vol_t(date)**2


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

    def simulate(self, fixings=None, corr = None, Nsim=1, seed=14,**kwargs):
        np.random.seed(seed)
        fixings = np.array(fixings)
        if fixings.shape==():
            fixings = np.array([fixings])
        Nsim = int(Nsim)
        if corr is None:
            print("Single Asset Simulation")
            logmartingale = np.zeros((int(2*Nsim),len(fixings)))
            for i in range (len(fixings)):
                Z = np.random.randn(Nsim)
                Z = np.concatenate((Z,-Z))
                if i ==0:
                    logmartingale.T[i]=-0.5*quad_piecewise(self.variance,self.variance.T,0,fixings[i])+sqrt(quad_piecewise(self.variance,self.variance.T,0,fixings[i]))*Z
                elif i!=0:
                    logmartingale.T[i]=logmartingale.T[i-1]-0.5*quad_piecewise(self.variance,self.variance.T,fixings[i-1],fixings[i])+sqrt(quad_piecewise(self.variance,self.variance.T,fixings[i-1],fixings[i]))*Z
            return exp(logmartingale)*self.forward_curve(fixings)

        else:
            print("Multi Asset Simulation")
            Ndim = len(corr)
            logmartingale = np.zeros((int(2*Nsim),len(fixings),Ndim))
            R = cholesky(corr)
            for i in range (len(fixings)):
                Z = np.random.randn(Nsim,Ndim)
                Z = np.concatenate((Z,-Z)) #matrix of uncorrelated random variables
                ep = np.dot(R,Z.T)   #matrix of correlated random variables
                for j in range(Ndim):
                    if i ==0:
                        logmartingale[:,i,j]=-0.5*quad_piecewise(self.variance[j],self.variance[j].T,0,fixings[i])+sqrt(quad_piecewise(self.variance[j],self.variance[j].T,0,fixings[i]))*ep[j]
                    elif i!=0:
                        logmartingale[:,i,j]=logmartingale[:,i-1,j]-0.5*quad_piecewise(self.variance[j],self.variance[j].T,fixings[i-1],fixings[i])+sqrt(quad_piecewise(self.variance[j],self.variance[j].T,fixings[i-1],fixings[i]))*ep[j]
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

def ACT_365(date1,date2):
    """Day count convention for a normal year"""
    return abs(date1-date2)/365

def ACT_360(date1,date2):
    """Day count convention for a 360 year"""
    return abs(date1-date2)/360


"""Definition of a piecewise function"""
def piecewise_function(date, interval, value):
    mask = np.array([])
    mask = np.append(mask,(date>=0) & (date<interval[0]))
    for i in range(1,len(interval)):
        mask= np.append(mask,(date>=interval[i-1]) & (date<interval[i]))
    y = 0
    for i in range(len(interval)):
        y = y+ value[i]*mask[i]
    y = y+value[len(value)-1]*(date>=interval[len(value)-1])  #from the last date to infinity I assume as value that one of the last intervall
    return y

def quad_piecewise(f, time_grid, t_in, t_fin, vectorial=0):
    """integral of a piecewise constant function"""
    dt = np.array([])
    t_in = float(t_in)
    t_fin = float(t_fin)
    time_grid=np.float64(time_grid)
    if t_in == t_fin:
        return 0
    if t_fin in time_grid:
        time_grid = time_grid[np.where(time_grid<=t_fin)[0]]
    if t_in in time_grid:
        time_grid = time_grid[np.where(time_grid>=t_in)[0]]
    if t_in not in time_grid:
        time_grid = time_grid[np.where(time_grid>t_in)[0]]
        time_grid = np.insert(time_grid,0,t_in)
    if t_fin not in time_grid:
        time_grid = time_grid[np.where(time_grid<t_fin)[0]]
        time_grid = np.insert(time_grid,len(time_grid),t_fin)
    if vectorial:
        y = np.array([])
        for i in range(1,len(time_grid)):
            y = np.append(y,f(time_grid[i]))
    else:
        y = f(time_grid[:-1])
      
    dt = np.diff(time_grid)
    return np.sum(y*dt)
