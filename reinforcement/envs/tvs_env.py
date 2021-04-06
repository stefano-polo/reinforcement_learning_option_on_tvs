import gym
from gym import spaces
from gym.utils import seeding
from numpy import exp, log, sqrt
from numpy.linalg import cholesky
import numpy as np
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, Black, ForwardVariance
from envs.pricing.closedforms import European_option_closed_form
from envs.pricing.targetvol import Drift, CholeskyTDependent, Strategy, TVSForwardCurve, TargetVolatilityStrategy
from envs.pricing.read_market import MarketDataReader
from envs.pricing.n_sphere import sign_renormalization

class TVS_environment(gym.Env):
    """Target volatility strategy Option environment"""
    def __init__(self, filename= "TVS_example.xml", spot_I = 100, target_volatility=0.1,strike_opt=100., maturity=1., constraint = "only_long", action_bound = 25/100, sum_long=None, sum_short=None):
        #Preparing Time grid for the RL agent
        self.constraint = constraint
        self.I_0 = spot_I
        self.strike_option = strike_opt
        self.target_vol = target_volatility
        self.T = maturity
        self.Nsim = 1e3
        self.current_time = 0.
        n_observations = 365
        self.time_index = 0
        self.simulation_index = 0
        months = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
        self.time_grid = np.cumsum(months)/n_observations
        self.time_grid = np.insert(self.time_grid,0,0.)
        #Loading Market data and preparing the BS model"""
        reader = MarketDataReader(filename)
        self.spot_prices = reader.get_spot_prices()
        self.correlation = reader.get_correlation()
        self.correlation_chole = cholesky(self.correlation)
        self.N_equity = len(self.correlation)
        self.D = reader.get_discounts()
        self.discount = self.D(maturity)
        self.F = reader.get_forward_curves()
        self.V = reader.get_volatilities()
        #Creating the objects for the TVS
        self.mu = Drift(forward_curves = self.F)
        self.nu = CholeskyTDependent(variance_curves = self.V, correlation = self.correlation)
        self.TVSF = TVSForwardCurve(reference=0.,vola_target = self.target_vol, spot_price = self.I_0, mu = self.mu, nu = self.nu, discounting_curve = self.D)
        self.model = Black(fixings=self.time_grid, variance_curve=self.V, forward_curve=self.F)
        self.integral_variance = np.cumsum(self.model.variance[:,1:],axis=1).T
        self.integral_variance_sqrt = sqrt(self.integral_variance)
        if self.constraint == 'long_short_limit' and (sum_long is None or sum_short is None):
            raise Exception("You should provide the sum limit for short and long position")
        if sum_long is not None and sum_short is not None:
            self.sum_long = sum_long
            self.sum_short = sum_short
        #Observation space and action space of the RL agent
        if self.constraint != "only_long":
            low_action = np.ones(self.N_equity)*(-abs(action_bound)) - 1e-6  
            high_action = np.ones(self.N_equity)*(abs(action_bound)) + 1e-6
        else:
            low_action = np.ones(self.N_equity)*1e-7
            high_action = np.ones(self.N_equity)

        self.action_space = spaces.Box(low = np.float32(low_action),high = np.float32(high_action))
        high = np.ones(self.N_equity)*2.5
        low_bound = np.append(-high,0.)
        high_bound = np.append(high,self.T+1./365.)
        self.observation_space = spaces.Box(low=np.float32(low_bound),high=np.float32(high_bound))


    def step(self, action):
        assert self.action_space.contains(action)
        #Modify action of the agent to satisfy constraint over the allocation strategy
        if self.constraint == "only_long":
            action = action/np.sum(action)
        elif self.constraint == "long_short_limit":
            action = sign_renormalization(action,self.sum_long,self.sum_short)

        if self.time_index == 0:
            #storing the rl agent's action
            self.alpha_t = action
        else:
            self.alpha_t = np.vstack([self.alpha_t, action])
        #evolving enviroment from t to t+1
        self.time_index = self.time_index + 1
        self.current_time = self.time_grid[self.time_index]
        self.current_asset = self.S_t[self.time_index]
        if self.current_time < self.T:
            #before the maturity the agent's reward is zero
            done = False
            reward = 0.
        else:
            #at maturity the agent collects its reward that is the discounted payoff of the TVS call option
            done = True
            alpha = Strategy(strategy = self.alpha_t, dates = self.time_grid[:-1])
            self.TVSF.set_strategy(alpha)
            TVS = TargetVolatilityStrategy(forward_curve=self.TVSF)
            I_t = TVS.simulate(fixings=np.array([self.T]), random_gen=self.np_random)[0,0]
            reward = np.maximum(I_t-self.strike_option,0.)*self.discount
            self.simulation_index =  self.simulation_index + 1

        #self.asset_history = np.append(self.asset_history,self.current_asset)
        state = np.append(self.current_asset, self.current_time)
        return state, reward, done, {}




    def reset(self):
        if self.simulation_index==0 or self.simulation_index == self.Nsim:
            #evolve the Black and Scholes model
            self.simulations = self.model.simulate(corr_chole=self.correlation_chole, random_gen=self.np_random, normalization=1, Nsim=self.Nsim)
            self.simulations[:,0,:]=0.
            self.simulations[:,1:,:] = (self.simulations[:,1:,:]+0.5*self.integral_variance)/self.integral_variance_sqrt
            self.simulation_index = 0
        self.current_time = 0.
        self.time_index = 0
        self.S_t = self.simulations[self.simulation_index]
        self.alpha_t = np.array([])
        state = np.append(self.S_t[self.time_index], self.current_time)
        return state


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def render(self, mode='human'):
        print()
        print('asset_history = ', self.asset_history)
        print('current time = ', self.current_time)


    def theoretical_price(self):
        s_righ = Strategy()
        if self.constraint == "only_long":
            s_right.optimization_constrained(mu=self.mu, nu=self.nu, N_trial= 50)
        else:
            s_right.Mark_strategy(mu = self.mu, nu = self.nu)
        TVSF = TVSForwardCurve(reference = 0, vola_target = self.target_vol, spot_price = self.I_0, strategy = s_right, mu = self.mu, nu = self.nu, discounting_curve = self.D)
        return European_option_closed_form(TVSF(self.T),self.strike_option,self.T,0,self.D.r(self.T),self.target_vol,1)
