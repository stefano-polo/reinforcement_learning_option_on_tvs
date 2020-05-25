import numpy as np
from scipy import exp, sqrt, log, heaviside
from pricing import piecewise_function, Curve, PricingModel
from numpy.linalg import cholesky


"""Notation of pallavicini and daluiso draft"""


class Drift(Curve):
    """Time dependent Repo rates union"""
    def __init__(self, forward_curves = None):
        self.T = time_grid_union(curve_array_list = forward_curves)
        Ndim = len(forward_curves)
        self.mu = np.zeros((len(self.T)+1,Ndim))
        for i in range(len(self.T)+1):
            for j in range(Ndim):
                if i ==0:
                    self.mu[i,j] = forward_curves[j].q[i]
                else:
                    self.mu[i,j] = (1./(forward_curves[j].reference-self.T[i-1])) * log((forward_curves[j](self.T[i-1])*forward_curves[j].discounting_curve(self.T[i-1]))/forward_curves[j].spot)  #repo rates
        self.mu = np.delete(self.mu, len(self.mu)-1,axis=0)
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
        for i in range(len(self.nu.T)):
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

    def optimal(self, mu = None, nu = None, Ntrials = 10, seed=14):
        Ntrials = int(Ntrials)
        np.random.seed(seed)
        Ndim = len(mu(0))
        self.T = np.array([])
        self.T = np.append(self.T,mu.T)
        self.T = np.append(self.T,nu.T)
        self.T = np.sort(np.asarray(list(set(self.T))))
        self.alpha_t = np.zeros((len(self.T)+1,Ndim))   #time dependent allocation strategy
        for i in range(len(self.T)+1):
            alpha = np.random.uniform(0,1,(Ntrials,Ndim ))
            norm = np.sum(alpha,axis=1)       #normalization
            alpha = (alpha.T/norm).T              #allocation strategy matrix (sum along each row is 1)
            if i==0:
                f = np.dot(alpha,mu(0.))/np.linalg.norm(np.dot(alpha,nu(0.)),axis=1)  #score function vector
            else:
                d= np.linalg.norm(np.dot(alpha,nu(self.T[i-1])),axis=1)
                if np.sum(d)==0:   #if divide by zero
                    f = 0
                else:
                    f = np.dot(alpha,mu(self.T[i-1]))/d
            self.alpha_t[i] = alpha[np.argmin(f)]

        self.alpha_t = np.delete(self.alpha_t, len(self.alpha_t)-1,axis=0)
        print("Optimal strategy time grid :",self.T)
        print("Optimal strategy : ",self.alpha_t)


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
        phi = piecewise_function(date-self.reference,self.T,self.phi)
        omega_t = self.vol/np.linalg.norm(np.dot(self.alpha(date),self.nu(date)))
        l = omega_t * np.dot(self.alpha(date),self.mu(date))
        return (self.I_0/self.D(date)) * exp(-(phi+l)*(date-self.reference))


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


class TargetVolatilityEuler(PricingModel):
    """Simulate the TVS price process with the Euler Method"""
    def __init__(self, reference = None, vola_target = None, spot_price = None, strategy = None, mu = None,nu = None, discounting_curve = None, fees = None, fees_dates = None):
        self.reference = 0
        self.vol = vola_target
        self.alpha = strategy
        self.I_0 = spot_price
        self.mu = mu
        self.nu = nu
        self.phi = fees
        self.T = fees_dates
        self.D = discounting_curve

    def simulate(self, fixings=None, Nsim=1, seed=14,**kwargs):
        np.random.seed(seed)
        Nsim = int(Nsim)
        Ndim = int(len(self.nu(0)))
        I_t = np.zeros((2*Nsim,len(fixings)))
        for i in range (len(fixings)):
            Z = np.random.normal(0,1,(Nsim,Ndim))
            Z = np.concatenate((Z,-Z))
            t = i-1
            r = piecewise_function(fixings[t],self.D.T,self.D.r)
            phi = piecewise_function(fixings[t],self.T,self.phi)

            if i == 0:
                omega = (self.vol*self.alpha(0.))/np.linalg.norm(np.dot(self.alpha(0.),self.nu(0.)))
                I_t[:,i] = self.I_0+ self.I_0 * (r-phi-np.dot(omega,self.mu(0.)))*(fixings[i]) + self.I_0*sqrt(fixings[i])*np.dot(np.dot(omega,self.nu(0.)),Z.T)

            else:
                omega = (self.vol*self.alpha(fixings[t]))/np.linalg.norm(np.dot(self.alpha(fixings[t]),self.nu(fixings[t])))
                I_t[:,i] = I_t[:,i-1]+ I_t[:,i-1] * (r-phi-np.dot(omega,self.mu(fixings[t])))*(fixings[i]-fixings[i-1]) + I_t[:,i-1]*sqrt(fixings[i]-fixings[i-1])*np.dot(np.dot(omega,self.nu(fixings[t])),Z.T)
        return I_t
