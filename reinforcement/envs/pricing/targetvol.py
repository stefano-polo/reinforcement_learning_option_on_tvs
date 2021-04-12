import numpy as np
from numpy import exp, sqrt, log
from envs.pricing.pricing import piecewise_function, Curve, PricingModel, quad_piecewise
from numpy.linalg import cholesky
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.interpolate import interp1d


"""Notation of pallavicini and daluiso draft"""


class Drift(Curve):
    """Time dependent Repo rates union"""
    def __init__(self, forward_curves = None):
        self.T = time_grid_union(curve_array_list = forward_curves)
        Ndim = len(forward_curves)
        self.mu = forward_curves[0].q(self.T)
        self.mu = np.stack((self.mu,forward_curves[1].q(self.T)),axis=1)
        """Calculating the instant repo rates"""
        for i in range(2,Ndim):
            self.mu = np.insert(self.mu,len(self.mu.T),forward_curves[i].q(self.T),axis=1)
        self.m = interp1d(self.T, self.mu, axis=0, kind='previous',fill_value="extrapolate", assume_sorted=False) 
       # print("Drift time grid:",self.T)
        #print("Drift values:", self.mu)   
    def curve(self,date):
        return self.m(date)


class CholeskyTDependent(Curve):
    """Time dependent cholesky variance-covariance matrix union"""
    def __init__(self, variance_curves = None, correlation_chole = None):
        self.T = time_grid_union(curve_array_list = variance_curves)
        N_times = len(self.T)
        Ndim = len(variance_curves)
        self.nu = np.zeros((Ndim,Ndim,N_times))
        I = np.identity(Ndim)
        for i in range(N_times):
            vol = np.zeros(Ndim)
            for j in range(Ndim):
                vol[j] = sqrt(variance_curves[j](self.T[i]))
            vol = I*vol
            self.nu[:,:,i] = vol@correlation_chole
        self.n = interp1d(self.T, self.nu, axis=2, kind='previous',fill_value="extrapolate", assume_sorted=False) 
      #  print("Cholesky covariance-variance time grid:",self.T)
     #   print("Cholesky covariance-variance matrix values:", self.nu)
   
    def curve(self,date):
        return self.n(date)

        
class Strategy(Curve):
    """Create the time dependent optimal strategy for BS model"""
    def __init__(self, strategy = None, dates = None):
        self.alpha_t = strategy
        self.T = dates
        if strategy is not None:
            self.a_t = interp1d(self.T, self.alpha_t, axis=0, kind='previous',fill_value="extrapolate", assume_sorted=False)
        
    def Mark_strategy(self,mu = None, nu = None):
        Ndim = len(mu(0.))
        self.T = np.union1d(mu.T,nu.T)
        self.alpha_t = np.zeros((len(self.T),Ndim))   #time dependent allocation strategy
        for i in range(len(self.T)):
            a_plus = Markowitz_solution(mu(self.T[i]),nu(self.T[i]),1)
            a_minus = Markowitz_solution(mu(self.T[i]),nu(self.T[i]),-1)
            if loss_function(a_plus,mu(self.T[i]),nu(self.T[i]))> loss_function(a_minus,mu(self.T[i]),nu(self.T[i])):
                self.alpha_t[i] = a_minus
            else:
                self.alpha_t[i] = a_plus
        
        self.a_t = interp1d(self.T, self.alpha_t, axis=0, kind='previous',fill_value="extrapolate", assume_sorted=False)
       # print("Markowitz strategy time grid :",self.T)
        #print("Markowitz strategy : ",self.alpha_t)

    def optimization_constrained(self, mu = None, nu = None, long_limit = 25/100, short_limit = 25/100, N_trial = 20, seed = 13, typo = 1):
        Ndim = len(mu(0.))
        self.T = np.union1d(mu.T,nu.T)
        if np.max(mu.T)>np.max(nu.T):    #check control to avoid denominator divergence
            self.T = self.T[np.where(self.T<=np.max(nu.T))[0]]
        self.alpha_t = np.zeros((len(self.T),Ndim))   #time dependent allocation strategy
        for i in range(len(self.T)):
            if typo == 1:
                    result = optimization_only_long(mu(self.T[i]), nu(self.T[i]),N_trial,seed)
            elif typo == 2:
                result = optimization_limit_position(mu(self.T[i]), nu(self.T[i]), long_limit,N_trial,seed)
            else:
                result = optimization_long_short_position(mu(self.T[i]), nu(self.T[i]), long_limit, short_limit,N_trial,seed)
            self.alpha_t[i] = result
        self.a_t = interp1d(self.T, self.alpha_t, axis=0, kind='previous',fill_value="extrapolate", assume_sorted=False)
       # print("Optimal strategy time grid :",self.T)
       # print("Optimal strategy through minimization: ",self.alpha_t)


    def Intuitive_strategy1(self, forward_curves=None, maturity_date=None):
        """Invest all on the asset with maximum growth at maturity"""
        asset_index = Max_forward_maturity(forward_curves=forward_curves,maturity=maturity_date)
        self.T = np.array([0,maturity_date])
        self.alpha_t = np.zeros((2,len(forward_curves)))
        self.alpha_t[:,asset_index] = 1
        print("Strategy time grid: ",self.T)
        print("Intuitive strategy (invest all on the asset with maximum growth at maturity) ",self.alpha_t)

    def Intuitive_strategy2(self,mu=None):
        """Invest all on the asset with minimum mu parameter"""
        asset_index = Min_mu_each_time(drift=mu)
        self.T = mu.T
        self.alpha_t = np.zeros((len(asset_index),len(mu(np.array([0.])))))
        for j in range (len(asset_index)):
            self.alpha_t[j,int(asset_index[j])] = 1
        print("Strategy time grid: ",self.T)
        print("Intuitive strategy (invest all on the asset with minimum mu parameter)",self.alpha_t)

    def Intuitive_strategy3(self,mu=None,nu=None):
        """Invest all on the asset with minimum mu/nu variable"""
        asset_index,self.T = Min_mu_nu_each_time(drift=mu,nu=nu)
        self.alpha_t = np.zeros((len(asset_index),len(mu(np.array([0.])))))
        for j in range (len(asset_index)):
            self.alpha_t[j,int(asset_index[j])] = 1
        print("Strategy time grid: ",self.T)
        print("Intuitive strategy (invest all on the asset with minimum mu/||nu|| parameter)",self.alpha_t)

    def curve(self, date):
        return self.a_t(date)



class TVSForwardCurve(Curve):

    def __init__(self, reference = 0, vola_target = None, spot_price = None, mu = None,nu = None, discounting_curve = None, fees = None, fees_dates = None):
        self.reference = reference
        self.vol = vola_target     #target volatility
        self.I_0 = spot_price
        self.mu = mu
        self.nu = nu
        self.D = discounting_curve
        
    def set_strategy(self,strategy=None):
        self.alpha=strategy
        
    def curve(self,date):
        date = np.array(date)
        l = lambda x: self.vol*np.sum(self.alpha(x)*self.mu(x),axis=1)/np.linalg.norm(np.sum((self.alpha(x).T*self.nu(x).transpose(1,0,2)),axis=1).T,axis=1)
        if date.shape!=():
            return np.asarray([(self.I_0/self.D(extreme)) * exp(-quad_piecewise(l,self.alpha.T,self.reference,extreme,vectorial=0)) for extreme in date])
        else:
            return (self.I_0/self.D(date)) * exp(-quad_piecewise(l,self.alpha.T,self.reference,date,vectorial=0))




class TargetVolatilityStrategy(PricingModel):
    """Simulation based on the exact solution of the SDE of the TVS price process"""
    def __init__(self, forward_curve = None):
        self.forward = forward_curve
        self.alpha = self.forward.alpha
        self.nu = self.forward.nu
        self.vol = self.forward.vol      #target volatility

    def simulate(self, fixings=None, Nsim=1, random_gen=None, ret_forward = 0, **kwargs):
        Nsim = int(Nsim)
        Ndim = int(len(self.nu(0.)))
        N_times = len(fixings)
        logI = np.zeros((Nsim,N_times))
        for i in range(N_times):
            Z = random_gen.randn(Nsim,Ndim)
            if i ==0:
                prod = self.alpha(0.)@self.nu(0.)
                omega_t = self.vol/np.linalg.norm(prod)
                logI[:,i] = -0.5*(self.vol**2)*fixings[i]+sqrt(fixings[i])*((omega_t*prod)@Z.T)
            else:
                prod = self.alpha(fixings[i-1])@self.nu(fixings[i-1])
                omega_t = (self.vol)/np.linalg.norm(prod)
                dt = fixings[i]-fixings[i-1]
                logI[:,i] = logI[:,i-1] -0.5*(self.vol**2)*(dt)+sqrt(dt)*((omega_t*prod)@Z.T)

        I =  np.zeros((Nsim,N_times))
        if ret_forward == 0:
            I = exp(logI)*self.forward(fixings)
            return I
        else:
            forward_curve = self.forward(fixings)
            I = exp(logI)*forward_curve
            return I, forward_curve


def time_grid_union(curve_array_list = None):
    """Create a unique temporal structure for calculation of optimal strategy"""
    T = curve_array_list[0].T     #union of all the temporal grids
    for i in range(1,len(curve_array_list)):
        T = np.union1d(T,curve_array_list[i].T)
    return T


def Max_forward_maturity(forward_curves=None,maturity=None):
    Ndim = len(forward_curves)
    F = np.zeros(Ndim)
    for i in range(Ndim):
        F[i] = forward_curves[i](maturity)
    return np.argmax(F)

def Min_mu_each_time(drift=None):
    n_times = len(drift.T)
    asset = np.zeros(n_times)
    for i in range (n_times):
        if i == 0:
            asset[i] = np.argmin(drift(0.))
        else:
            asset[i] = np.argmin(drift(drift.T[i-1]))
    return asset

def Min_mu_nu_each_time(drift=None,nu=None):
    T = drift.T
    T = np.union1d(T,nu.T)
    n_times = len(T)
    asset = np.zeros(n_times)
    for i in range (n_times):
        if i == 0:
            asset[i] = np.argmin(drift(0.)/np.linalg.norm(nu(0.)))
        else:
            asset[i] = np.argmin(drift(T[i-1])/np.linalg.norm(nu(T[i-1])))
    return asset, T

def Markowitz_solution(mu,nu,sign):
    """Closed form of the optimal strategy"""
    S = nu@nu.T  #covariance matrix
    S_1 = np.linalg.inv(S) #inverse of covariance matrix
    prod = S_1@mu
    p_m = prod@nu
    norm = sqrt(p_m @ p_m)
    normalization = sign*0.5*norm
    return 0.5*(1./normalization)*(prod)

def loss_function(x,mu,nu):
    """Target function to minimize"""
    prod = x@nu
    norm = sqrt(prod@prod)
    return (x@mu)/norm

def optimization_only_long(mu=None, nu=None,seed = 1, N_trial=None, guess = None):
    """Constrained optimization with only long position and sum of weights equal to 1"""
    if guess is None and N_trial is None:
        N_trial = 1
    f = loss_function
    # cons = ({'type': 'eq','fun' : lambda x: np.sum(x)-1},{'type': 'ineq','fun' : lambda x: x})
    A = np.ones(len(mu))
    x_low = np.array([1.])
    x_up = np.array([1.])
    bounds = Bounds(np.zeros(len(mu)),np.ones(len(mu)))
    cons = LinearConstraint(A,x_low,x_up)
    if guess is None and N_trial is not None:
        np.random.seed(seed)
        r = np.zeros((N_trial,len(mu)))
        valutation = np.zeros(N_trial)
        for i in range (N_trial):
            x0 =np.random.uniform(0.,1.,len(mu))  #initial position for the optimization algorithm
            res = minimize(f, x0, args=(mu,nu),constraints=cons,bounds=bounds,method="SLSQP")#,options={'ftol': 1e-30})
            r[i] = res.x
            valutation[i] = f(res.x,mu,nu)
        #print("Minumum: ", np.min(valutation))
        return r[np.argmin(valutation)]
    elif guess.ndim==1:
        return minimize(f, guess, args=(mu,nu),constraints=cons,bounds = bounds, method = "SLSQP").x#,options={'ftol': 1e-30})
    elif guess.ndim>1:
        N_trial = len(guess)
        r = np.zeros((N_trial,len(mu)))
        valutation = np.zeros(N_trial)
        for i in range(N_trial):
            x0 = guess[i]
            res =  minimize(f, x0, args=(mu,nu),constraints=cons,bounds=bounds,method="SLSQP")
            r[i] = res.x
            valutation[i] = f(res.x,mu,nu)
        return r[np.argmin(valutation)]

def optimization_limit_position(mu, nu, limit_position,N_trial=3,seed=None):
    """Constrained optimization with each |weight|<limit_position"""
    np.random.seed(seed)
    f = loss_function
    cons = ({'type': 'ineq','fun' : lambda x: -abs(x)+limit_position})
    r = np.zeros((N_trial,len(mu)))
    valutation = np.zeros(N_trial)
    for i in range (N_trial):
        x0 =np.random.uniform(-limit_position,limit_position,len(mu))
        res = minimize(f, x0, args=(mu,nu),constraints=cons)
        r[i] = res.x
        valutation[i] = f(res.x,mu,nu)
    #print("Minumum: ", np.min(valutation))
    return r[np.argmin(valutation)]

def optimization_long_short_position(mu, nu, long_limit, short_limit,N_trial,seed):
    """Constrained optimization with each limit on maximum """
    np.random.seed(seed)
    f = loss_function
    cons = ({'type': 'ineq','fun' : lambda x: -np.sum(x[np.where(x>0)[0]])+long_limit},{'type': 'ineq','fun' : lambda x: -abs(np.sum(x[np.where(x<0)[0]]))+short_limit})
    r = np.zeros((N_trial,len(mu)))
    valutation = np.zeros(N_trial)
    for i in range (N_trial):
        x0 =np.random.uniform(-short_limit,long_limit,len(mu))
        res = minimize(f, x0, args=(mu,nu),constraints=cons)
        r[i] = res.x
        valutation[i] = f(res.x,mu,nu)
    #print("Minumum: ", np.min(valutation))
    return r[np.argmin(valutation)]
