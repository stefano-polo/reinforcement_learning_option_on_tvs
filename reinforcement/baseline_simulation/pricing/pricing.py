import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from numpy import exp, sqrt, log
from scipy.integrate import quad
import matplotlib.pyplot as plt

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
        if self.T[0] == 0.:
            self.T = np.delete(self.T,0)
            repo_rates = np.delete(repo_rates,0)
        self.q_values = np.array([repo_rates[0]])
        for i in range(1,len(self.T)):
            alpha = ((self.T[i])*(repo_rates[i])-(self.T[i-1])*
                     (repo_rates[i-1]))/(self.T[i]-self.T[i-1])
            self.q_values = np.append(self.q_values,alpha)
        self.T = np.append(0.,self.T[:-1])
        self.q = interp1d(self.T, self.q_values, kind='previous',fill_value="extrapolate", assume_sorted=False) 
     #   print("Forward repo time grid",self.T)
      #  print("Forward repo rate: ", self.q_values)

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
            r_instant = np.array([(-1./self.T[1])*log(discounts[1])])
            r_instant = np.append(r_instant,(-1./(self.T[2:]-self.T[1:-1]))*log(discounts[2:]/discounts[1:-1]))
        else:
            r_zero = (-1./(self.T))*log(discounts)
        self.R = interp1d(self.T,r_zero) #zero rate from 0 to T1
        self.r_t = interp1d(self.T[:-1],r_instant,kind='previous',fill_value="extrapolate")  #instant interest rate
      #  print("zero interest rate time grid",self.T)
      #  print("zero interest rate: ",r_zero)

    def curve(self, date):
        return exp(-self.R(date)*date)


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
            self.original = self.T
            self.spot_vol = np.array([])
            self.matrix = spot_volatility
            self.K = strikes
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
     #   print("Forward volatility time grid: ",self.T)
     #   print("Forward volatility: ",self.forward_vol)

    def curve(self,date):
        return self.vol_t(date)**2

class LocalVolatilityCurve():
    
    def __init__(self, volatility_parameters=None, moneyness_matrix=None, maturities=None, asset_name=None):
        self.name = asset_name
        self.lv = volatility_parameters
        self.log_moneyness = np.log(moneyness_matrix)
        self.T = maturities
        n_dates = len(maturities)     #it is fundamental this transformation for the piecewise interpolation
        time_idx = tuple(range(n_dates))
        if n_dates>1:
            self.time_interpolator = interp1d(maturities, time_idx, kind='next', fill_value='extrapolate')
        else:
            self.time_interpolator = lambda t: 0
    
    def time_indices_simulation(self, time_indexes):
        self.interpolator_strikes = []
        self.money_max = np.array([])
        self.money_min = np.array([])
        self.LV_max = np.array([])
        self.LV_min = np.array([])
        for i in range(time_indexes[-1]+1):
            this_money = self.log_moneyness[:,i]
            this_lv    = self.lv[:,i]
            self.interpolator_strikes.append(PchipInterpolator(this_money, this_lv))
            self.money_max = np.append(self.money_max,np.max(this_money))
            self.money_min = np.append(self.money_min,np.min(this_money))
            self.LV_min = np.append(self.LV_min,this_lv[0])
            self.LV_max = np.append(self.LV_max,this_lv[-1])
            
        
    def intelligent_call(self,index,k):
        eta = self.interpolator_strikes[index](k)
        eta[k<self.money_min[index]] = self.LV_min[index]
        eta[k>self.money_max[index]] = self.LV_max[index]
        return eta 
    
    def __call__(self,t,k):
        idx = int(self.time_interpolator(t))
        this_money = self.log_moneyness[:,idx]
        this_lv    = self.lv[:,idx]
        eta =  interp1d(this_money, this_lv, kind='nearest', fill_value='extrapolate')(k)#PchipInterpolator(this_money, this_lv)(k)
        eta[k<this_money[0]] = this_lv[0]
        eta[k>this_money[-1]] = this_lv[-1]
        return eta

class PricingModel:

    def __init__(self, **kwargs):

      raise Exception('model not implemented.')

    def simulate(self, fixings=None, Nsim=1, seed=14, **kwargs):

      raise Exception('simulate not implemented.')


class Black(PricingModel):
    """Black model"""

    def __init__(self, fixings=None, variance_curve=None, forward_curve=None,**kwargs):
        if type(forward_curve) == list:
            N_equity = len(forward_curve)
            N_times = len(fixings)
            self.forward = np.zeros((N_equity,N_times))
            for i in range(N_equity):
                self.forward[i,:] = forward_curve[i](fixings)
            self.variance = np.zeros((N_equity,N_times))
            for i in range(N_equity):
                for j in range(N_times):
                    if j==0:
                        self.variance[i,j] = quad_piecewise(variance_curve[i],variance_curve[i].T,0.,fixings[j])
                    else:
                        self.variance[i,j] = quad_piecewise(variance_curve[i],variance_curve[i].T,fixings[j-1],fixings[j])
        else:
            N_times = len(fixings)
            self.forward = forward_curve(fixings)
            self.variance = np.zeros(N_times)
            for j in range(N_times):
                if j==0:
                    self.variance[j] = quad_piecewise(variance_curve,variance_curve.T,0.,fixings[j])
                else:
                    self.variance[j] = quad_piecewise(variance_curve,variance_curve.T,fixings[j-1],fixings[j])

    def simulate(self, random_gen = None, corr_chole = None, Nsim=1, seed=14, normalization=1,**kwargs):
        Nsim = int(Nsim)
        N_times = len(self.variance.T)
        if corr_chole is None:
            logmartingale = np.zeros((Nsim,N_times))
            for i in range (N_times):
                Z = random_gen.randn(Nsim)
                if i ==0:
                    logmartingale.T[i]=-0.5*self.variance[i]+sqrt(self.variance[i])*Z
                elif i!=0:
                    logmartingale.T[i]=logmartingale.T[i-1]-0.5*self.variance[i]+sqrt(self.variance[i])*Z
            return exp(logmartingale)*self.forward

        else:
            Ndim = len(corr_chole)
            logmartingale = np.zeros((Nsim,N_times,Ndim))
            for i in range (N_times):
                Z = np.random.randn(Nsim,Ndim)
                ep = corr_chole@Z.T   #matrix of correlated random variables
                for j in range(Ndim):
                    if i ==0:
                        logmartingale[:,i,j]=-0.5*self.variance[j,i]+sqrt(self.variance[j,i])*ep[j]
                    elif i!=0:
                        logmartingale[:,i,j]=logmartingale[:,i-1,j]-0.5*self.variance[j,i]+sqrt(self.variance[j,i])*ep[j]
            if normalization:
                return logmartingale
            else:    
                M = exp(logmartingale)*self.forward.T
                return M


class LV_model(PricingModel):
    """Local Volatility Model"""
    def __init__(self, fixings=None, local_vol_curve=None, forward_curve=None, N_grid = 100,**kwargs):
        self.vol = local_vol_curve
        if fixings[0] == 0.:
            fixings = fixings[1:]
        self.time_grid, self.dt = Eulero_grid(fixings,N_grid)
        self.N_grid = N_grid
        if type(forward_curve) == list:
            """Multiasset LV model"""
            self.Ndim = len(forward_curve)
            N_times = len(fixings)
            self.forward = np.zeros((self.Ndim,len(self.time_grid)))
            self.time_indexes_matrix = np.array([])
            for i in range(self.Ndim):
                self.forward[i,:] = forward_curve[i](self.time_grid)
                time_indexes = self.vol[i].time_interpolator(self.time_grid).astype(int)
                self.vol[i].time_indices_simulation(time_indexes)
                if i == 0:
                    self.time_indexes_matrix = time_indexes
                else:
                    self.time_indexes_matrix = np.vstack([self.time_indexes_matrix,time_indexes])
        else:
            """One asset LV model"""
            self.forward = forward_curve(fixings)
            self.time_indexes = self.vol.time_interpolator(self.time_grid).astype(int)
            self.vol.time_indices_simulation(self.time_indexes)
            
    def simulate(self, random_gen = None, corr_chole = None, Nsim=1, normalization=1,**kwargs):
        Nsim = int(Nsim)
        N_times = len(self.time_grid)
        N_fixings = int(len(self.time_grid)/self.N_grid)
        counter = 1
        time_index = 0
        dt = self.dt[time_index]
        if corr_chole is None:
            logmartingale = np.zeros((int(Nsim),N_fixings))
            logX = 0.
            for i in range (N_times):
                Z = random_gen.randn(Nsim)
                if i ==0:
                    vol = self.vol.intelligent_call(0,logX)
                    logX= logX-0.5*dt*(vol**2)+vol*sqrt(dt)*Z
                elif i!=0:
                    vol = self.vol.intelligent_call(self.time_indexes[i-1],logX)
                    logX = logX-0.5*dt*(vol**2)+vol*sqrt(dt)*Z
                counter = i+1
                if  counter%self.N_grid == 0:
                    logmartingale[:,time_index] = logX
                    time_index +=1
                    if counter < N_times:
                        dt = self.dt[time_index]
            if normalization:
                return logmartingale
            else:
                return exp(logmartingale)*self.forward
        else:
           # logmartingale = np.zeros((Nsim.N_fixings,self.Ndim))
            logX = np.zeros((Nsim,self.Ndim))
           # correlated_wiener = np.array([])
            logmartingale = np.array([])
            vola_t = np.array([])
            for i in range (N_times):
                Z = random_gen.randn(Nsim,self.Ndim)
                ep = corr_chole@Z.T   #matrix of correlated random variables
               # correlated_wiener = np.append(correlated_wiener,ep)
                for j in range(self.Ndim):
                    if i ==0:
                        vol = self.vol[j].intelligent_call(0,0.)
                        #print("Asset "+str(j+1)+"vola ",vol)
                        vola_t = np.append(vola_t,vol*np.ones(Nsim))
                        logX[:,j]=-0.5*dt*(vol**2)+vol*sqrt(dt)*ep[j]
                        logmartingale = np.append(logmartingale,logX[:,j])
                    elif i!=0:
                        vol = self.vol[j].intelligent_call(self.time_indexes_matrix[j,i-1],logX[:,j])
                        vola_t = np.append(vola_t,vol)
                        #print("Asset "+str(j+1)+"vola ",vol)
                        logX[:,j]=logX[:,j]-0.5*dt*(vol**2)+vol*sqrt(dt)*ep[j]
                        logmartingale = np.append(logmartingale,logX[:,j])
                counter = i+1
                if counter%self.N_grid == 0:
                    #logmartingale[:,time_index,:] = logX
                    time_index +=1
                    if counter < N_times:
                        dt = self.dt[time_index]
            if normalization:
                return (correlated_wiener.reshape(N_times,self.Ndim,Nsim)).transpose(2,0,1), (vola_t.reshape(N_times,self.Ndim,Nsim)).transpose(2,0,1)
            else:    
                logmartingale = (logmartingale.reshape(N_times,self.Ndim,Nsim)).transpose(2,0,1)
                M = exp(logmartingale)*self.forward.T
                return M, (vola_t.reshape(N_times,self.Ndim,Nsim)).transpose(2,0,1)
      
"""Payoff Functions"""
def Vanilla_PayOff(St=None,strike=None, typo = 1): #Monte Carlo call payoff
    zero = np.zeros(St.shape)
    if typo ==1:
        """Call option payoff"""
        pay = St-strike
    elif typo ==-1:
        """Put option payoff"""
        pay = strike - St
   # pay1,pay2 = np.split(np.maximum(pay,zero),2) #for antithetic sampling
    
    return np.maximum(pay,zero)#0.5*(pay1+pay2)

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

def Eulero_grid(fixings=None,N_intervals=None):
    time_grid = np.array([])
    for i in range(len(fixings)):
        if i == 0:
            dt = np.array([fixings[i]/N_intervals])
            time_grid = np.append(time_grid, np.linspace(dt[i],fixings[i],N_intervals))
        else:
            dt = np.append(dt,(fixings[i]-fixings[i-1])/N_intervals)
            time_grid = np.append(time_grid,np.linspace(fixings[i-1]+dt[i],fixings[i],N_intervals))
    return time_grid, dt
