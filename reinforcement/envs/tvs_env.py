import gym
from gym import spaces
from gym.utils import seeding
from numpy import exp, log, sqrt
import numpy as np
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, Black, ForwardVariance
from envs.pricing.closedforms import European_option_closed_form
from envs.pricing.targetvol import Drift, CholeskyTDependent, Strategy, TVSForwardCurve, TargetVolatilityStrategy
from envs.pricing.read_market import MarketDataReader


class TVS_enviroment(gym.Env):
    """Target volatility Option environment"""
    def __init__(self, =None, spot_I = 100, target_volatility=0.2,strike_opt=100., maturity=1.):
        """Loading Market data and preparing the BS model"""
        self.reader = MarketDataReader(filename)
        self.N_equity = reader.get_stock_number() - 6
        self.spot_prices = reader.get_spot_prices()
        self.correlation_matrix = reader.get_correlation()
        self.D = reader.get_discounts()
        self.F = reader.get_forward_curves()
        self.V = reader.get_volatilities()
        self.model = Black(forward_curve = self.F, variance = self.V)
        """Creating the objects for the TVS"""
        self.mu = Drift(forward_curves = self.F)
        self.nu = CholeskyTDependent(variance_curves = self.V, correlation = self.correlation_matrix)
        """Preparing Time grid for the RL agent"""
        self.I_0 = spot_I
        self.strike_option = strike_opt
        self.current_time = 0
        self.T = maturity
        self.time_index = 0
        self.time_grid = np.linspace(0,self.T,12)   #the agent observe the enviroment each month
        self.asset_history = self.spot_prices
        """Observation space and action space of the RL agent"""
        
        low_action = np.ones(self.N_equity-1)*1e-8   #the agent can choose the asset allocation strategy only for N-1 equities (the N one is set by 1-sum(weights_of_other_equities))
        high_action = np.ones(self.N_equity-1)
        self.action_space = spaces.Box(low = low_action, high = high_action)
        low_bound = np.zeros(1+self.N_equity)                      #the observation space is the prices space plus the time space
        high_bound = np.append(self.spot_prices*10000,self.T+1/365)
        self.observation_space = spaces.Box(low=low_bound,high=high_bound)
        self.seed()
        self.reset()


    def step(self, action):  # metodo che mi dice come evolve il sistema una volta arrivata una certa azione
        assert self.action_space.contains(action)   #gli arriva una certa azione

        #if current time is = 0 evolve the prices of the equities along all the time grid of the simulation
        if self.current_time == 0:
            self.asset_history = self.model.simulate(fixings=self.time_grid, Ndim= self.N_equity, corr = self.correlation_matrix, random_gen = self.np_random)[0]

        if self.current_time < self.T:
        #for t<T the agent oberve the universe of assets and its actions (allocation strategy) are stored along all the simulation
            if self.time_index == 0:
                self.alpha_t = action
            else:
                self.alpha_t = np.vstack([self.alpha_t, action])
            state = np.append(self.current_asset, self.current_time)
            done = False
            reward = 0.
            self.time_index = self.time_index+1
            self.current_time = self.time_grid[self.time_index]
            self.current_asset = self.asset_history[self.time_index]
            return state, reward, done, {}

      ##### in t = T the agent collect its reward
        if self.current_time == self.T:
            done = True
            self.current_asset = self.S_t[self.time_index-1]
            self.current_time = self.time_grid[self.time_index-1]
            state = np.append(self.current_asset, self.current_time)
            alpha = Strategy(strategy = self.alpha_t, dates = self.time_grid[:-1])
            TVSF = TVSForwardCurve(reference = self.reference, vola_target = self.target_vol, spot_price = self.spot_I, strategy = alpha, mu = self.mu, nu = self.nu, discounting_curve = self.D)
            TVS = TargetVolatilityStrategy(forward_curve=TVSF)
            I_t = TVS.simulate(fixings=array([self.T]), random_gen=self.np_random)[0,0]
            reward = np.maximum(I_t-self.strike_opt,0)*self.D(self.T)
            return state, reward, done, {}


    def reset(self):
        self.current_time = 0.
        self.time_index = 0
        self.alpha_t  = np.array([])
        self.asset_history = self.spot_prices
        self.current_asset = self.spot_prices
        state = np.append(self.current_asset, self.current_time)
        return state


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def render(self, mode='human'):
        print()
        print('asset_history = ', self.asset_history)
        print('current time = ', self.current_time)


    def theoretical_price(self):
        return European_option_closed_form(forward = self.s0*exp(self.r*self.t1), strike= self.strike_opt, maturity=self.t1, reference=0, zero_interest_rate = self.r, volatility=self.sigma, typo=1)
