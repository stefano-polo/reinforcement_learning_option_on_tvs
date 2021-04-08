import gym
from gym import spaces
from gym.utils import seeding
from numpy import log, sqrt, exp
import numpy as np
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, LV_model, ForwardVariance, quad_piecewise
from envs.pricing.closedforms import European_option_closed_form
from envs.pricing.targetvol import Drift, Strategy, TVSForwardCurve, TargetVolatilityStrategy, optimization_only_long
from envs.pricing.fake_market_lv import load_fake_market_lv
from envs.pricing.loadfromtxt import LoadFromTxt
from envs.pricing.targetvol import optimization_only_long, Markowitz_solution
from envs.pricing.n_sphere import sign_renormalization

class TVS_LV(gym.Env):
    """Target volatility strategy Option environment with a Local volatility model for the assets
    """
    def __init__(self, N_equity= 2, frequency = "month", target_volatility=5/100, I_0 = 1., strike_opt=1., 
    maturity=2., constraint = "free", action_bound=5., sum_long = None, sum_short=None):
        """Pricing parameters for the option"""
        self.bang_bang_action = 0
        self.baseline = 1
        self.constraint = constraint
        self.target_vol = target_volatility
        self.I_0 = I_0
        self.I_t = I_0
        self.strike_opt = strike_opt
        self.N_equity = N_equity            
        self.T = maturity
        names = ["DJ 50 EURO E","S&P 500 NET EUR"]
        correlation = np.array(([1.,0.],[0.,1.]))
        folder_name = "FakeSmiles"
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
        if self.baseline:
            mu_function = Drift(forward_curves=F)
            self.mu_values  = mu_function(np.append(0.,euler_grid[:-1]))
        self.r_t = D.r_t(np.append(0.,euler_grid[:-1]))
        self.discount = D(self.T)
        self.dt_vector = self.model.dt
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
        if not self.bang_bang_action:
            if not self.baseline:
                if self.constraint == 'long_short_limit' and (sum_long is None or sum_short is None):
                    raise Exception("You should provide the sum limit for short and long position")
                if sum_long is not None and sum_short is not None:
                    self.sum_long = sum_long
                    self.sum_short = sum_short
                if self.constraint != "only_long":
                    low_action = np.ones(self.N_equity)*(-abs(action_bound))-1e-6
                    high_action = np.ones(self.N_equity)*abs(action_bound)+1e-6
                else:
                    low_action = np.ones(self.N_equity)*1e-7
                    high_action = np.ones(self.N_equity)
            else:
                low_action = np.ones(self.N_equity)*(-1.)
                high_action = np.ones(self.N_equity)
            self.action_space = spaces.Box(low = np.float32(low_action), high = np.float32(high_action))
        else:
            self.action_space = spaces.Discrete(2)
        high = np.ones(N_equity)*2.5
        low_bound = np.append(-high,0.)
        high_bound = np.append(high,self.T+1./365)
        self.observation_space = spaces.Box(low=np.float32(low_bound),high=np.float32(high_bound))


#current time start at zero
    def step(self, action): 
        assert self.action_space.contains(action)
        if not self.bang_bang_action:
            if self.constraint == "only_long" and not self.baseline:
                action = action/np.sum(action)
                s = 1.
            elif self.constraint == "long_short_limit":
                action = sign_renormalization(action,self.how_long,self.how_short)
                s = np.sum(action)
            elif self.constraint == "free" and not self.baseline:
                s = np.sum(action)
        else:
            action = np.array([action, 1.-action])
            s = 1.
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
            if self.baseline and i==0:
                if self.constraint == "only_long":
                    baseline = optimization_only_long(self.mu_values[idx], nu,seed=1, guess = np.array(([0.4,0.6],[0.6,0.4])))        
                    action = action + baseline
                    action[action<0] = 0.
                    if action.any() == np.zeros(self.N_equity).any():
                        action = baseline
                    action = action/np.sum(action)
                    s = 1.
                elif self.constraint == "free":
                    baseline = Markowitz_solution(self.mu_values[idx],nu,-1)
                    action = action+baseline
                    s = np.sum(action)                
            omega = self.target_vol/np.linalg.norm(action@nu)
            self.I_t = self.I_t * (1. + omega * action@self.dS_S[idx] + dt * self.r_t[idx]*(1.-omega*s))
        if self.current_time < self.T:
            done = False
            reward = 0.
        else:
            done = True
            reward = np.maximum(self.I_t-self.strike_opt,0.)*self.discount
            self.simulation_index = self.simulation_index +1

        state = np.append(self.current_logX, self.current_time)
        return state, reward, done, {}


    def reset(self):
        if self.simulation_index == 0 or self.simulation_index==self.Nsim:
            self.simulations_logX = None
            self.simulations_Vola = None
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
        self.logX_t = self.simulations_logX[0]
        self.dS_S = self.dS_S_simulations[0]
        self.sigma_t = self.simulations_Vola[0]
        self.simulations_Vola = np.delete(self.simulations_Vola,0,axis=0)
        self.dS_S_simulations = np.delete(self.dS_S_simulations,0,axis=0)
        self.simulations_logX = np.delete(self.simulations_logX,0,axis=0)
        state = np.append(self.logX_t[0], self.current_time)
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