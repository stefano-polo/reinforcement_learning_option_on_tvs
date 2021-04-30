import gym
import envs.GemSessionUtility as gsu
import pyGem as gem
from gym import spaces
from gym.utils import seeding
from numpy import log, sqrt, exp
import numpy as np
from scipy.integrate import quad
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, LV_model, ForwardVariance, quad_piecewise
from envs.pricing.targetvol import Drift, Strategy, TVSForwardCurve, CholeskyTDependent, optimization_only_long
from envs.pricing.loadfromtxt import LoadFromTxt
from envs.pricing.targetvol import optimization_only_long, Markowitz_solution
from scipy.interpolate import interp1d

class TVS_LV_newreward(gym.Env):
    """Target volatility strategy Option environment with a Local volatility model for the assets
    """
    def __init__(self, N_equity= 2, frequency = "month", target_volatility=5/100, I_0 = 1., strike_opt=1., 
    maturity=2., constraint = "only_long", action_bound=5., sum_long = None, sum_short=None):
        #Simulation parameters
        self.constraint = constraint
        
        self.target_vol = target_volatility
        self.N_equity = N_equity            
        self.strike_opt = strike_opt
        self.I_0 = I_0
        self.I_t = I_0
        self.T = maturity
        self.V_t = 0. #derivative value at time t
        self.V_t_plus = 0. #derivative value at time t+1

        names = ["DJ 50 EURO E","S&P 500 NET EUR"]
        correlation = np.array(([1.,0.6],[0.6,1.]))
        folder_name = "FakeSmilesDisplacedDiffusion"
        ACT = 365.
        """Time grid creation for the simulation"""
        self.Identity = np.identity(self.N_equity)
        if frequency == "month":
            month_dates = np.array([31.,28.,31.,30.,31.,30.,31.,31.,30.,31.,30.,31.])
            months = month_dates
            if self.T > 1.:
                for i in range(int(self.T)-1):
                    months = np.append(months, month_dates)
            self.observation_grid = np.cumsum(months)/ACT
            self.N_euler_grid = 60       
            self.state_index = np.arange(int(12*self.T)+1)*self.N_euler_grid
        elif frequency == "day":
            self.observation_grid = np.linspace(1./ACT, self.T, int(self.T*ACT))
            self.N_euler_grid = 2
            self.state_index = np.arange(int(365*self.T)+1)*self.N_euler_grid
        
        self.observation_grid = np.append(0.,self.observation_grid)
        self.time_index = 0
        self.current_time = 0.
        self.simulation_index = 0
        self.Nsim = 1e3
        """Loading market curves"""
        D, F, V, LV = LoadFromTxt(names, folder_name)
        self.correlation_chole = np.linalg.cholesky(correlation)
        self.spot_prices = np.zeros(self.N_equity)
        for i in range(self.N_equity):
            self.spot_prices[i] = F[i].spot
        """Preparing the LV model"""
        self.model = LV_model(fixings=self.observation_grid[1:], local_vol_curve=LV, forward_curve=F, N_grid = self.N_euler_grid)
        euler_grid = self.model.time_grid
        """Discount part"""
        self.r_t = D.r_t(np.append(0.,euler_grid[:-1]))
        self.discount = D(self.T)
        self.intermidiate_discounts = D(self.T)/D(self.observation_grid)
        self.dt_vector = self.model.dt
        """Hedging costs elements"""
        self.mu_function = Drift(forward_curves=F)
        self.mu_grid = np.append(self.mu_function.T[self.mu_function.T<self.T],self.T)
        """Black variance for normalization"""
        self.integral_variance = np.zeros((N_equity,len(self.observation_grid[1:])))
        for i in range(self.N_equity):
            for j in range(len(self.observation_grid[1:])):
                self.integral_variance[i,j] = quad_piecewise(V[i],V[i].T,0.,self.observation_grid[j+1])
        self.integral_variance = self.integral_variance.T
        self.integral_variance = np.insert(self.integral_variance,0,np.zeros(self.N_equity),axis=0)
        self.integral_variance_sqrt =sqrt(self.integral_variance)
        self.integral_variance_sqrt[0,:] = 1
        self.forwards = np.insert(self.model.forward.T,0,self.spot_prices,axis=0)[self.state_index,:]
        if self.constraint == "free":
            low_action = np.ones(self.N_equity)*(-abs(action_bound))
            high_action = np.ones(self.N_equity)*abs(action_bound)
        elif self.constraint == "only_long":
            low_action = np.ones(self.N_equity)*1e-10
            high_action = np.ones(self.N_equity)
        self.action_space = spaces.Box(low = np.float32(low_action), high = np.float32(high_action))
        high = np.ones(N_equity)*2.5
        low_bound = np.append(-high,np.zeros(2))
        high_bound = np.append(high,np.array([self.I_0*10.,self.T+1./365]))
        self.observation_space = spaces.Box(low=np.float32(low_bound),high=np.float32(high_bound))


#current time start at zero
    def step(self, action): 
        assert self.action_space.contains(action)
        if self.constraint == "only_long":
            action = action/np.sum(action)
            s = 1.
        elif self.constraint == "free":
            s = np.sum(action)
        
        self.time_index = self.time_index + 1
        self.current_time = self.observation_grid[self.time_index]
        self.current_logX = self.logX_t[self.time_index]
        dt = self.dt_vector[self.time_index-1]
        index_plus = (self.time_index-1)*self.N_euler_grid
        """Simulation of I_t"""
        for i in range(self.N_euler_grid):
            idx = index_plus + i 
            Vola =  self.sigma_t[idx]*self.Identity
            nu = Vola@self.correlation_chole
            prod = action@nu
            norm = sqrt(prod@prod)               
            omega = self.target_vol/norm
            self.I_t = self.I_t * (1. + omega * action@self.dS_S[idx] + dt * self.r_t[idx]*(1.-omega*s))
        
        """Evaluation of the reward"""
        if self.constraint=="free":
            a_bs = Markowitz_solution(self.mu_function(self.current_time),nu,-1)
        else:
            a_bs = np.zeros(self.N_equity) 
            a_bs[np.argmax(self.sigma_t[idx])] = 1. 
        prod = a_bs@nu
        norm = sqrt(prod@prod)
        omega = self.target_vol/norm
        mask = self.mu_grid>self.current_time
        grid = np.append(self.current_time,self.mu_grid[mask])
        mu_values = self.mu_function(grid[1:])
        local_drift = np.sum(np.sum(a_bs *mu_values,axis=1)*np.diff(grid))*omega
        forward_tvs = self.I_t * (np.exp(-local_drift)/self.intermidiate_discounts[self.time_index])
        self.V_t_plus = gem.GemUtility.BS(self.T-self.current_time,forward_tvs,self.strike_opt,self.target_vol,1,self.intermidiate_discounts[self.time_index])[0]
        reward = self.V_t_plus - self.V_t
        
        if self.current_time < self.T:
            done = False
            self.V_t = self.V_t_plus
        else:
            done = True
            self.simulation_index = self.simulation_index + 1 

           
        state = np.append(self.current_logX, np.array([self.I_t,self.current_time]))
        print(state)
        return state, reward, done, {}


    def reset(self):
        if self.simulation_index == 0 or self.simulation_index==self.Nsim:
            self.simulations_logX = None
            self.simulations_Vola = None
            self.dS_S_simulations = None
            S, self.simulations_Vola = self.model.simulate(corr_chole = self.correlation_chole, random_gen = self.np_random, normalization = 0, Nsim=self.Nsim)
            S = np.insert(S,0,self.spot_prices,axis=1)
            self.dS_S_simulations = (S[:,1:,:] - S[:,:-1,:])/S[:,:-1,:]
            S_sliced = S[:,self.state_index,:]
            self.simulations_logX = (log(S_sliced/self.forwards)+0.5*self.integral_variance)/self.integral_variance_sqrt
            S_sliced = None
            S = None
            self.simulation_index=0
        self.current_time = 0.
        self.time_index = 0
        self.I_t = self.I_0
        self.logX_t = self.simulations_logX[self.simulation_index]
        self.dS_S = self.dS_S_simulations[self.simulation_index]
        self.sigma_t = self.simulations_Vola[self.simulation_index]
        state = np.append(self.logX_t[0], np.array([self.I_0,self.current_time]))
        self.V_t = self.V_t_plus = 0
        return state


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        print()
        print('asset_history = ', self.asset_history)
        print('current time = ', self.current_time)


    def theoretical_price(self):
        return 0
