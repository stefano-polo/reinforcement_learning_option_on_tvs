import gym
from gym import spaces
from gym.utils import seeding
from numpy import log, sqrt, exp
import numpy as np
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, Black, ForwardVariance
from envs.pricing.closedforms import European_option_closed_form
from envs.pricing.targetvol import Drift, CholeskyTDependent, Strategy, TVSForwardCurve, TargetVolatilityStrategy
from envs.pricing.read_market import MarketDataReader
from envs.pricing.fake_market import load_fake_market
from envs.pricing.n_sphere import n_sphere_to_cartesian

class TVS_simple(gym.Env):
    """Target volatility strategy Option environment with a simple market
    """
    def __init__(self, N_equity= 2, target_volatility=5/100, I_0 = 1., r=1/100., strike_opt=1., maturity=1., constraint = "only_long", action_bound=50, sum_long = None, sum_short=None):
        self.constraint = constraint
        self.target_vol = target_volatility
        self.spot_I = I_0
        self.N_equity = N_equity                                #number of equities
        self.asset_history = np.array([])
        self.T = maturity
        n_observations = 12
        self.time_grid = np.linspace(self.T/n_observations,self.T,n_observations)
        self.time_grid = np.insert(self.time_grid,0,0)
        self.time_index = 0
        self.current_time = 0.
        self.asset_history = np.array([])
        self.alpha_t = np.array([])
        self.strike_opt = strike_opt
        self.simulation_index = 0
        self.Nsim = 1e4
        self.D, self.F, self.V, self.correlation, self.spot_prices = load_fake_market(N_equity, r, self.T)
        self.discount = self.D(self.T)
        self.correlation_chole = np.linalg.cholesky(self.correlation)
        self.model = Black(fixings=self.time_grid,variance_curve=self.V,forward_curve = self.F)
        self.mu = Drift(self.F)
        self.nu = CholeskyTDependent(self.V,self.correlation)
        self.vola_t = sqrt(np.sum(self.nu(self.time_grid)**2,axis=1)).T
        if self.constraint == 'long_short_limit' and (sum_long is None or sum_short is None):
            raise Exception("You should provide the sum limit for short and long position")
        if sum_long is not None and sum_short is not None:
            self.sum_long = sum_long
            self.sum_short = sum_short
        if self.constraint != "only_long":
            low_action = np.ones(self.N_equity)*(-abs(action_bound))
            high_action = np.ones(self.N_equity)*abs(action_bound)
        else:
            low_action = np.zeros(self.N_equity-1)
            high_action = np.ones(self.N_equity-1)*(0.5*np.pi+0.005)
        self.action_space = spaces.Box(low = np.float32(low_action), high = np.float32(high_action))
        high = np.ones(N_equity)*2.5
        low_bound = np.append(-high,0.)
        high_bound = np.append(high,self.T+1./365)
        self.observation_space = spaces.Box(low=np.float32(low_bound),high=np.float32(high_bound))


#current time start at zero
    def step(self, action):  # metodo che mi dice come evolve il sistema una volta arrivata una certa azione
        assert self.action_space.contains(action)   #gli arriva una certa azione
        if self.constraint == "only_long":
            action = n_sphere_to_cartesian(1,action)**2
        elif self.constraint == "long_short_limit":
            action = sign_renormalization(action,self.how_long,self.how_short)

        if self.time_index == 0:
            self.alpha_t = action
        else:
            self.alpha_t = np.vstack([self.alpha_t, action])
        self.time_index = self.time_index+1
        self.current_time = self.time_grid[self.time_index]
        self.current_asset = self.S_t[self.time_index]

        if self.current_time < self.T:
            done = False
            reward = 0.
        else:
            done = True
            alpha = Strategy(strategy = self.alpha_t, dates = self.time_grid[:-1])
            TVSF = TVSForwardCurve(reference = 0, vola_target = self.target_vol, spot_price = self.spot_I, strategy = alpha, mu = self.mu, nu = self.nu, discounting_curve = self.D)
            TVS = TargetVolatilityStrategy(forward_curve=TVSF)
            I_t = TVS.simulate(fixings=np.array([self.T]), random_gen=self.np_random)[0,0]
            reward = np.maximum(I_t-self.strike_opt,0.)*self.discount
            self.simulation_index = self.simulation_index +1

        state = np.append(self.current_asset, self.current_time)
        return state, reward, done, {}


    def reset(self):
        if self.simulation_index == 0 or self.simulation_index==self.Nsim:
            self.simulations = self.model.simulate(corr_chole = self.correlation_chole, random_gen = self.np_random, Nsim=self.Nsim)
            #self.simulations[:,1:,:] = (log(self.simulations[:,1:,:]/self.simulations[:,:-1,:])-0.5*self.model.variance.T[1:])/sqrt(self.model.variance.T[1:])
            #self.simulations[:,0,:] = 0.
            self.simulations = log(self.simulations/self.spot_prices)/self.vola_t
            self.simulation_index=0
        self.current_time = 0.
        self.S_t = self.simulations[self.simulation_index]
        self.alpha_t = np.array([])
        self.time_index = 0
        self.asset_history = np.array([])
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
        s_right.Mark_strategy(mu = self.mu, nu = self.nu)
        TVSF = TVSForwardCurve(reference = self.reference, vola_target = self.target_vol, spot_price = self.spot_I, strategy = s_right, mu = self.mu, nu = self.nu, discounting_curve = self.D)
        return European_option_closed_form(TVSF(self.T),self.strike_option,self.T,0,self.r,self.target_vol,1)
