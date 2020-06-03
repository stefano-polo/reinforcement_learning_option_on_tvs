import numpy as np
from scipy import exp, sqrt, log, heaviside
from pricing import piecewise_function, Curve, PricingModel
from numpy.linalg import cholesky
from scipy.optimize import minimize
from scipy.integrate import quad
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
                 mu_zero[i,j] = (1/(forward_curves[j].reference-self.T[i]))*log((forward_curves[j](self.T[i])*forward_curves[j].discounting_curve(self.T[i]))/forward_curves[j].spot)

        """Calculating the forward repo rates"""
        self.mu = np.zeros((len(self.T),Ndim))
        for i in range(len(self.T)):
            for j in range(Ndim):
                if i==0:
                    self.mu[i][j] = mu_zero[i,j]
                else:
                    self.mu[i][j] = ((self.T[i]-forward_curves[j].reference)*(mu_zero[i][j])-(self.T[i-1]-forward_curves[j].reference)*(mu_zero[i-1][j]))/(self.T[i]-self.T[i-1])
        print("Drift time grid:",self.T)
        print("Drift values:", self.mu)

    def curve(self,date):
        return piecewise_function(date,self.T,self.mu)


class CholeskyTDependent(Curve):
    """Time dependent cholesky variance-covariance matrix union"""
    def __init__(self, variance_curves = None, correlation = None):
        self.T = time_grid_union(curve_array_list = variance_curves)
        Ndim = len(variance_curves)
        self.nu = np.zeros((Ndim,Ndim,len(self.T)+1))
        for i in range(len(self.T)+1):
            vol = np.zeros(Ndim)
            if i == 0:
                for j in range(Ndim):
                    vol[j] = variance_curves[j].forward_vol[i]
                vol = np.identity(Ndim)*vol
                self.nu[:,:,i] = np.dot(np.dot(vol,correlation),vol)
                
            else:
                for j in range(Ndim):
                    vol[j] = sqrt(variance_curves[j](self.T[i-1]))
                vol = np.identity(Ndim)*vol
                self.nu[:,:,i] = np.dot(np.dot(vol,correlation),vol)
                
        self.nu = np.delete(self.nu,len(self.nu.T)-1,axis=2)
        for i in range(len(self.T)):
            self.nu[:,:,i] = cholesky(self.nu[:,:,i])
        print("Cholesky covariance-variance time grid:",self.T)
        print("Cholesky covariance-variance matrix values:", self.nu)


    def curve(self,date):
        return piecewise_function(date,self.T,self.nu)

class Strategy(Curve):
    """Create the time dependent optimal strategy for BS model"""
    def __init__(self, strategy = None, dates = None):
        self.alpha_t = strategy
        self.T = dates

    def Mark_strategy(self,mu = None, nu = None):
        Ndim = len(mu(0))
        self.T = np.array([])
        self.T = np.append(self.T,mu.T)
        self.T = np.append(self.T,nu.T)
        self.T = np.sort(np.asarray(list(set(self.T))))
        self.alpha_t = np.zeros((len(self.T),Ndim))   #time dependent allocation strategy
        for i in range(len(self.T)):
            
            if i==0:
                Cov = np.dot(nu(0.),nu(0.).T)
                self.alpha_t[i] = Markowitz(mu(0.),Cov)  #score function vector
            else:
                Cov = np.dot(nu(self.T[i-1]),nu(self.T[i-1]).T)
                self.alpha_t[i] = Markowitz(mu(self.T[i-1]),Cov)  #score function vector

        print("Markowitz strategy time grid :",self.T)
        print("Markowitz strategy : ",self.alpha_t)
        
    def optimal(self, mu = None, nu = None, Ntrials = 10, seed=14):
        Ntrials = int(Ntrials)
        np.random.seed(seed)
        Ndim = len(mu(0))
        self.T = np.array([])
        self.T = np.append(self.T,mu.T)
        self.T = np.append(self.T,nu.T)
        self.T = np.sort(np.asarray(list(set(self.T))))
        self.alpha_t = np.zeros((len(self.T),Ndim))   #time dependent allocation strategy
        bnds = ((-4,4),(-4,4))    
        for i in range(len(self.T)):
            if i ==0:
                x0 = np.ones(Ndim)*0.2
                cons = ({'type': 'eq','fun' : lambda x: np.sum(x)-1})
                f = lambda x: np.dot(x,mu(0.))/np.linalg.norm(np.dot(x,nu(0.)))
                res = minimize(f, x0,constraints=cons)
            else:
                x0 = res.x
                cons = ({'type': 'eq','fun' : lambda x: np.sum(x)-1})
                f = lambda x: np.dot(x,mu(self.T[i-1]))/np.linalg.norm(np.dot(x,nu(self.T[i-1])))
                res = minimize(f, x0,constraints=cons)
            self.alpha_t[i] = res.x
        print("Optimal strategy time grid :",self.T)
        print("Optimal strategy through minimization: ",self.alpha_t)
    
        
    def Intuitive_strategy1(self, forward_curves=None, maturity_date=None):
        """Invest all on the asset with maximum growth at maturity"""
        asset = Max_forward_maturity(forward_curves=forward_curves,maturity=maturity_date)
        self.T = np.array([0,maturity_date])
        self.alpha_t = np.zeros((2,len(forward_curves)))
        self.alpha_t[:,asset] = 1
        print("Strategy time grid: ",self.T)
        print("Intuitive strategy (invest all on the asset with maximum growth at maturity ",self.alpha_t)
    
    def Intuitive_strategy2(self,mu=None): 
        """Invest all on the asset with minimum mu parameter"""
        asset = Min_mu_each_time(drift=mu)
        self.T = mu.T
        self.alpha_t = np.zeros((len(asset),len(mu(0.))))
        for j in range (len(asset)):
            self.alpha_t[j,int(asset[j])] = 1
        print("Strategy time grid: ",self.T)
        print("Intuitive strategy (invest all on the asset with minimum mu parameter",self.alpha_t)      
    
    def Intuitive_strategy3(self,mu=None,nu=None):
        """Invest all on the asset with minimum mu/nu variable"""
        asset,self.T = Min_mu_nu_each_time(drift=mu,nu=nu)
        self.alpha_t = np.zeros((len(asset),len(mu(0.))))
        for j in range (len(asset)):
            self.alpha_t[j,int(asset[j])] = 1
        print("Strategy time grid: ",self.T)
        print("Intuitive strategy (invest all on the asset with minimum mu/nu parameter",self.alpha_t)      
    
    def curve(self, date):
        return piecewise_function(date,self.T,self.alpha_t)



class TVSForwardCurve(Curve):

    def __init__(self, reference = None, vola_target = None, spot_price = None, strategy = None, mu = None,nu = None, discounting_curve = None, fees = None, fees_dates = None):
        self.reference = 0
        self.vol = vola_target     #target volatility
        self.alpha = strategy
        self.I_0 = spot_price
        self.mu = mu
        self.nu = nu
        self.phi = fees
        self.T = fees_dates
        self.D = discounting_curve

    def curve(self,date):
        phi = piecewise_function
        l = lambda x: self.vol*(np.dot(self.alpha(x),self.mu(x))/np.linalg.norm(np.dot(self.alpha(x),self.nu(x))))
        return (self.I_0/self.D(date)) * exp(-quad(l,self.reference,date)[0])*exp(-quad(phi,self.reference,date,args=(self.T,self.phi))[0])

    
 

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

    def simulate(self, fixings=None, Nsim=1, seed=14,**kwargs):
        Nsim = int(Nsim)
        np.random.seed(seed)
        Ndim = int(len(self.nu(0)))
        logI = np.zeros((2*Nsim,len(fixings)))
        for i in range(len(fixings)):
            Z = np.random.normal(0,1,(Nsim,Ndim))
            Z = np.concatenate((Z,-Z))
            if i ==0:
                omega_t = (self.vol)/np.linalg.norm(np.dot(self.alpha(0.),self.nu(0.)))
                logI[:,i] = -0.5*(np.linalg.norm(np.dot(omega_t*self.alpha(0.),self.nu(0.)))**2)*fixings[i]+sqrt(fixings[i])*np.dot(np.dot(omega_t*self.alpha(0.),self.nu(0.)),Z.T)
            else:
                omega_t = (self.vol)/np.linalg.norm(np.dot(self.alpha(fixings[i-1]),self.nu(fixings[i-1])))
                logI[:,i] = logI[:,i-1] -0.5*(np.linalg.norm(np.dot(omega_t*self.alpha(fixings[i-1]),self.nu(fixings[i-1])))**2)*(fixings[i]-fixings[i-1])+sqrt(fixings[i]-fixings[i-1])*np.dot(np.dot(omega_t*self.alpha(fixings[i-1]),self.nu(fixings[i-1])),Z.T)

        I =  np.zeros((2*Nsim,len(fixings)))
        for i in range (len(fixings)):
            I[:,i] = exp(logI[:,i])*self.forward(fixings[i])
        return I



def time_grid_union(curve_array_list = None):
    """Create a unique temporal structure for calculation of optimal strategy"""
    T = np.array([])     #union of all the temporal grids
    for i in range(len(curve_array_list)):
        T = np.append(T,curve_array_list[i].T)
    T = np.sort(np.asarray(list(set(T))))    #delete duplicate elements and sort the grid
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
    T = np.array([])
    T = np.append(T,drift.T)
    T = np.append(T,nu.T)
    T = np.sort(np.asarray(list(set(T))))
    n_times = len(T)
    asset = np.zeros(n_times)
    for i in range (n_times):
        if i == 0:
            asset[i] = np.argmin(drift(0.)/np.linalg.norm(nu(0.)))
        else:
            asset[i] = np.argmin(drift(T[i-1])/np.linalg.norm(nu(T[i-1])))
    return asset, T

def Markowitz(asset_return,covariance_matrix):
    W = np.linalg.inv(covariance_matrix)
    e = np.ones(len(asset_return))
    numerator = np.dot(W,(-asset_return))
    denominator = np.dot(np.dot(e.T,W),(-asset_return))
    return numerator/denominator