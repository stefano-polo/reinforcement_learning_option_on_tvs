import gym
from gym import spaces
from gym.utils import seeding
from numpy import exp, log, sqrt
import numpy as np
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, Black, ForwardVariance
from envs.pricing.closedforms import European_option_closed_form
from envs.pricing.targetvol import Drift, CholeskyTDependent, Strategy, TVSForwardCurve, TargetVolatilityStrategy
from envs.pricing.read_market import MarketDataReader
from envs.pricing.n_sphere import n_sphere_to_cartesian, sign_renormalization

class TVS_enviroment(gym.Env):
    """Target volatility strategy Option environment"""
    def __init__(self, filename= "TVS_example.xml", spot_I = 100, target_volatility=0.1,strike_opt=100., maturity=1., constraint = "free", action_bound = 25/100, sum_long=None, sum_short=None):
        #Loading Market data and preparing the BS model"""
        reader = MarketDataReader(filename)
        self.spot_prices = reader.get_spot_prices()
        self.correlation_matrix = reader.get_correlation()
        self.N_equity = len(self.correlation_matrix)
        self.D = reader.get_discounts()
        self.F = reader.get_forward_curves()
        self.V = reader.get_volatilities()
        self.model = Black(forward_curve = self.F, variance = self.V)
        #Creating the objects for the TVS
        self.mu = Drift(forward_curves = self.F)
        self.nu = CholeskyTDependent(variance_curves = self.V, correlation = self.correlation_matrix)
        self.vola_t = sqrt(np.sum(self.nu(0)**2,axis=0))
        for time in self.time_grid[1:]:
            self.vola_t =  np.vstack([self.vola_t, sqrt(np.sum(self.nu(time)**2,axis=0))])
        #Preparing Time grid for the RL agent
        self.I_0 = spot_I
        self.strike_option = strike_opt
        self.target_vol = target_volatility
        self.current_time = 0
        self.T = maturity
        self.time_index = 0
        self.time_grid = np.linspace(0,self.T,12)   #the agent observe the enviroment each month
        self.asset_history = np.array([])
        self.constraint = constraint
        if self.constraint == 'long_short_limit' and (sum_long is None or sum_short is None):
            raise Exception("You should provide the sum limit for short and long position")
        if sum_long is not None and sum_short is not None:
            self.sum_long = sum_long
            self.sum_short = sum_short
        #Observation space and action space of the RL agent
        if self.constraint != "only_long":
            low_action = np.ones(self.N_equity)*(-abs(action_bound))   #the agent can choose the asset allocation strategy only for N-1 equities (the N one is set by 1-sum(weights_of_other_equities))
            high_action = np.ones(self.N_equity)*(abs(action_bound))
        else:
            low_action = np.zeros(self.N_equity-1)
            high_action = np.ones(self.N_equity-1)*(np.pi*0.5+0.001)

        self.action_space = spaces.Box(low = np.float32(low_action),high = np.float32(high_action))
        high = np.ones(N_equity)*2.5
        low_bound = np.append(-high,0)                      #the observation space is the prices space plus the time space
        high_bound = np.append(high,self.T+1/365)
        self.observation_space = spaces.Box(low=np.float32(low_bound),high=np.float32(high_bound))
        self.seed()
        self.reset()


    def step(self, action):
        assert self.action_space.contains(action)
        #Modify action of the agent to satisfy constraint over the allocation strategy
        if self.constraint == "only_long":
            action = n_sphere_to_cartesian(1,action)**2
        elif self.constraint == "long_short_limit":
            action = sign_renormalization(action,self.sum_long,self.sum_short)

        if self.time_index == 0:
            #evolve the Black and Scholes model
            self.S_t = log(self.model.simulate(fixings=self.time_grid, corr = self.correlation_matrix, random_gen = self.np_random)[0]/self.spot_prices)/self.vola_t
            #storing the rl agent's action
            self.alpha_t = action
        else:
            self.alpha_t = np.vstack([self.alpha_t, action])
        #evolving enviroment from t to t+1
        self.time_index = self.time_index+1
        self.current_time = self.time_grid[self.time_index]
        self.current_asset = self.S_t[self.time_index]
        if self.current_time < self.T:
            #before the maturity the agent's reward is zero
            done = False
            reward = 0.
        else:
            #at maturity the agent collects its reward that is the discounted payoff of the TVS call option
            done = True
            alpha = Strategy(strategy = self.alpha_t, dates = self.time_grid[1:])
            TVSF = TVSForwardCurve(reference = 0, vola_target = self.target_vol, spot_price = self.I_0, strategy = alpha, mu = self.mu, nu = self.nu, discounting_curve = self.D)
            TVS = TargetVolatilityStrategy(forward_curve=TVSF)
            I_t = TVS.simulate(fixings=np.array([self.T]), random_gen=self.np_random)[0,0]
            reward = np.maximum(I_t-self.strike_option,0)*self.D(self.T)

        self.asset_history = np.append(self.asset_history,self.current_asset)
        state = np.append(self.current_asset, self.current_time)
        return state, reward, done, {}




    def reset(self):
        self.current_time = 0.
        self.time_index = 0
        self.alpha_t  = np.array([])
        #self.asset_history = self.spot_prices
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
        s_righ = Strategy()
        if self.constraint == "only_long":
            s_right.optimization_constrained(mu=self.mu, nu=self.nu, N_trial= 50)
        else:
            s_right.Mark_strategy(mu = self.mu, nu = self.nu)
        TVSF = TVSForwardCurve(reference = 0, vola_target = self.target_vol, spot_price = self.I_0, strategy = s_right, mu = self.mu, nu = self.nu, discounting_curve = self.D)
        return European_option_closed_form(TVSF(self.T),self.strike_option,self.T,0,self.D.r(self.T),self.target_vol,1)
