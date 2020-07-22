import numpy as np
from numpy import exp, sqrt, log, array
from envs.pricing.pricing import piecewise_function, Curve, PricingModel, quad_piecewise
from numpy.linalg import cholesky
from scipy.optimize import minimize


"""Notation of pallavicini and daluiso draft"""


class Drift(Curve):
    """Time dependent Repo rates union"""
    def __init__(self, forward_curves = None):
        self.T = time_grid_union(curve_array_list = forward_curves)
        Ndim = len(forward_curves)
        mu_zero = np.zeros((len(self.T),Ndim))
        """Calculating the zero repo rates"""
        for i in range(len(self.T)):
            for j in range(Ndim):
                if i ==0:
                    mu_zero[i,j] = forward_curves[j].q[0]
                else:
                    mu_zero[i,j] = (1/(-self.T[i]))*log((forward_curves[j](self.T[i])*forward_curves[j].discounting_curve(self.T[i]))/forward_curves[j].spot)

        """Calculating the forward repo rates"""
        self.mu = np.zeros((len(self.T),Ndim))
        for i in range(len(self.T)):
            for j in range(Ndim):
                if i==0:
                    self.mu[i][j] = mu_zero[i,j]
                else:
                    self.mu[i][j] = ((self.T[i])*(mu_zero[i][j])-(self.T[i-1])*(mu_zero[i-1][j]))/(self.T[i]-self.T[i-1])

    def curve(self,date):
        return piecewise_function(date,self.T,self.mu)


class CholeskyTDependent(Curve):
    """Time dependent cholesky variance-covariance matrix union"""
    def __init__(self, variance_curves = None, correlation = None):
        self.T = time_grid_union(curve_array_list = variance_curves)
        Ndim = len(variance_curves)
        self.nu = np.zeros((Ndim,Ndim,len(self.T)))
        for i in range(len(self.T)):
            vol = np.zeros(Ndim)
            if i == 0:
                for j in range(Ndim):
                    vol[j] = sqrt(variance_curves[j](0.))
                vol = np.identity(Ndim)*vol
                self.nu[:,:,i] = cholesky(vol@(correlation@vol))
            else:
                for j in range(Ndim):
                    vol[j] = sqrt(variance_curves[j](self.T[i-1]))
                vol = np.identity(Ndim)*vol
                self.nu[:,:,i] = cholesky(vol@(correlation@vol))


    def curve(self,date):
        return piecewise_function(date,self.T,self.nu.T)

class Strategy(Curve):
    """Create the time dependent optimal strategy for BS model"""
    def __init__(self, strategy = None, dates = None):
        self.alpha_t = strategy
        self.T = dates

    def Mark_strategy(self,mu = None, nu = None):
        Ndim = len(mu(0))
        self.T = np.union1d(mu.T,nu.T)
        self.alpha_t = np.zeros((len(self.T),Ndim))   #time dependent allocation strategy
        for i in range(len(self.T)):
            if i==0:
                a_plus = Markowitz_solution(mu(0.),nu(0.),1)
                a_minus = Markowitz_solution(mu(0.),nu(0.),-1)
                if loss_function(a_plus,mu(0.),nu(0.))> loss_function(a_minus,mu(0.),nu(0.)):
                    self.alpha_t[i] = a_minus
                else:
                    self.alpha_t[i] = a_plus
            else:
                a_plus = Markowitz_solution(mu(self.T[i-1]),nu(self.T[i-1]),1)
                a_minus = Markowitz_solution(mu(self.T[i-1]),nu(self.T[i-1]),-1)
                if loss_function(a_plus,mu(self.T[i-1]),nu(self.T[i-1]))> loss_function(a_minus,mu(self.T[i-1]),nu(self.T[i-1])):
                    self.alpha_t[i] = a_minus
                else:
                    self.alpha_t[i] = a_plus

        print("Markowitz strategy time grid :",self.T)
        print("Markowitz strategy : ",self.alpha_t)

    def optimization_constrained(self, mu = None, nu = None, long_limit = 25/100, short_limit = 25/100, N_trial = 20, seed = 13, typo = 1):
        Ndim = len(mu(0))
        self.T = np.union1d(mu.T,nu.T)
        if np.max(mu.T)>np.max(nu.T):    #check control to avoid denominator divergence
            self.T = self.T[np.where(self.T<=np.max(nu.T))[0]]
        self.alpha_t = np.zeros((len(self.T),Ndim))   #time dependent allocation strategy
        for i in range(len(self.T)):
            if i ==0:
                if typo == 1:
                    result = optimization_only_long(mu(0.), nu(0.),N_trial,seed)
                elif typo == 2:
                    result = optimization_limit_position(mu(0.), nu(0.), long_limit,N_trial,seed)
                else:
                    result = optimization_long_short_position(mu(0.), nu(0.), long_limit, short_limit,N_trial,seed)
            else:
                if typo == 1:
                    result = optimization_only_long(mu(self.T[i-1]), nu(self.T[i-1]),N_trial,seed)
                elif typo == 2:
                    result = optimization_limit_position(mu(self.T[i-1]), nu(self.T[i-1]), long_limit,N_trial,seed)
                else:
                    result = optimization_long_short_position(mu(self.T[i-1]), nu(self.T[i-1]), long_limit, short_limit,N_trial,seed)

            self.alpha_t[i] = result
        print("Optimal strategy time grid :",self.T)
        print("Optimal strategy through minimization: ",self.alpha_t)


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
        self.alpha_t = np.zeros((len(asset_index),len(mu(0.))))
        for j in range (len(asset_index)):
            self.alpha_t[j,int(asset_index[j])] = 1
        print("Strategy time grid: ",self.T)
        print("Intuitive strategy (invest all on the asset with minimum mu parameter)",self.alpha_t)

    def Intuitive_strategy3(self,mu=None,nu=None):
        """Invest all on the asset with minimum mu/nu variable"""
        asset_index,self.T = Min_mu_nu_each_time(drift=mu,nu=nu)
        self.alpha_t = np.zeros((len(asset_index),len(mu(0.))))
        for j in range (len(asset_index)):
            self.alpha_t[j,int(asset_index[j])] = 1
        print("Strategy time grid: ",self.T)
        print("Intuitive strategy (invest all on the asset with minimum mu/||nu|| parameter)",self.alpha_t)

    def curve(self, date):
        return piecewise_function(date,self.T,self.alpha_t)



class TVSForwardCurve(Curve):

    def __init__(self, reference = 0, vola_target = None, spot_price = None, strategy = None, mu = None,nu = None, discounting_curve = None, fees = array([0,0]), fees_dates = array([1,100])):
        self.reference = reference
        self.vol = vola_target     #target volatility
        self.alpha = strategy
        self.I_0 = spot_price
        self.mu = mu
        self.nu = nu
        self.phi = fees
        self.T = fees_dates
        self.D = discounting_curve

    def curve(self,date):
        date = np.array(date)
        phi = lambda x: piecewise_function(x,self.T,self.phi)
        l = lambda x: self.vol*((self.alpha(x)@self.mu(x))/np.linalg.norm(self.alpha(x)@self.nu(x)))
        if date.shape!=():
            return np.asarray([(self.I_0/self.D(extreme)) * exp(-quad_piecewise(l,self.alpha.T,self.reference,extreme))*exp(-quad_piecewise(phi,self.T,self.reference,extreme)) for extreme in date])
        else:
            return (self.I_0/self.D(date)) * exp(-quad_piecewise(l,self.alpha.T,self.reference,date))*exp(-quad_piecewise(phi,self.T,self.reference,date))




class TargetVolatilityStrategy(PricingModel):
    """Simulation based on the exact solution of the SDE of the TVS price process"""
    def __init__(self, forward_curve = None):
        self.forward = forward_curve
        self.alpha = self.forward.alpha
        self.nu = self.forward.nu
        self.mu = self.forward.mu
        self.I_0 = self.forward.I_0
        self.D = self.forward.D
        self.vol = self.forward.vol      #target volatility

    def simulate(self, fixings=None, Nsim=1, random_gen=None, ret_forward = 0, **kwargs):
        Nsim = int(Nsim)
        Ndim = int(len(self.nu(0)))
        logI = np.zeros((Nsim,len(fixings)))
        for i in range(len(fixings)):
            Z = random_gen.randn(Nsim,Ndim)
            if i ==0:
                omega_t = (self.vol)/np.linalg.norm(self.alpha(0.)@self.nu(0.))
                logI[:,i] = -0.5*(np.linalg.norm(omega_t*(self.alpha(0.)@self.nu(0.)))**2)*fixings[i]+sqrt(fixings[i])*((omega_t*self.alpha(0.)@self.nu(0.))@Z.T)
            else:
                omega_t = (self.vol)/np.linalg.norm(self.alpha(fixings[i-1])@self.nu(fixings[i-1]))
                logI[:,i] = logI[:,i-1] -0.5*(np.linalg.norm(omega_t*(self.alpha(fixings[i-1])@self.nu(fixings[i-1])))**2)*(fixings[i]-fixings[i-1])+sqrt(fixings[i]-fixings[i-1])*((omega_t*self.alpha(fixings[i-1])@self.nu(fixings[i-1]))@Z.T)

        I =  np.zeros((2*Nsim,len(fixings)))
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
    norm = sign*0.5*np.linalg.norm((S_1@mu)@nu)
    return 0.5*(1/norm)*(S_1@mu)

def loss_function(x,mu,nu):
    """Target function to minimize"""
    return (x@mu)/np.linalg.norm(x@nu)

def optimization_only_long(mu, nu, N_trial,seed):
    """Constrained optimization with only long position and sum of weights equal to 1"""
    np.random.seed(seed)
    f = loss_function
    cons = ({'type': 'eq','fun' : lambda x: np.sum(x)-1},{'type': 'ineq','fun' : lambda x: x})
    r = np.zeros((N_trial,len(mu)))
    valutation = np.zeros(N_trial)
    for i in range (N_trial):
        x0 =np.random.uniform(0.,1.,len(mu))  #initial position for the optimization algorithm
        res = minimize(f, x0, args=(mu,nu),constraints=cons)
        r[i] = res.x
        valutation[i] = f(res.x,mu,nu)
    #print("Minumum: ", np.min(valutation))
    return r[np.argmin(valutation)]

def optimization_limit_position(mu, nu, limit_position,N_trial,seed):
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
