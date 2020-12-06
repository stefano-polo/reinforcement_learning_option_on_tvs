import gym
from gym import spaces
from gym.utils import seeding
from numpy import log, sqrt, exp
import numpy as np
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, LV_model, ForwardVariance, quad_piecewise
from envs.pricing.closedforms import European_option_closed_form
from envs.pricing.targetvol import Drift, Strategy, TVSForwardCurve, TargetVolatilityStrategy
from envs.pricing.fake_market_lv import load_fake_market_lv
from envs.pricing.n_sphere import sign_renormalization

class TVS_LV(gym.Env):
    """Target volatility strategy Option environment with a simple market
    """
    def __init__(self, N_equity= 2, target_volatility=5/100, I_0 = 1., r=1/100., strike_opt=1., maturity=1., constraint = "only_long", action_bound=20./100., sum_long = None, sum_short=None):
        self.constraint = constraint
        self.target_vol = target_volatility
        self.I_0 = I_0
        self.I_t = I_0
        self.N_equity = N_equity                                #number of equities
        self.Identity = np.identity(self.N_equity)
        self.T = maturity
        n_observations = 365
        self.time_grid = np.linspace(self.T/n_observations,self.T,n_observations)
        self.time_grid = np.insert(self.time_grid,0,0.)
        self.dt = np.diff(self.time_grid)[0]
        self.time_index = 0
        self.current_time = 0.
        self.asset_history = np.array([])
        self.strike_opt = strike_opt
        self.simulation_index = 0
        self.Nsim = 1e3
        D, F, V, self.correlation = load_fake_market_lv()
        self.r_values = D.r(self.time_grid[:-1])
        self.discount = D(self.T)
        self.correlation_chole = np.linalg.cholesky(self.correlation)
        self.model = LV_model(fixings=self.time_grid, local_vol_curve=V, forward_curve=F)
        mu = Drift(F)
        self.mu_values = mu(self.time_grid[:-1])
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
        self.action_space = spaces.Box(low = np.float32(low_action), high = np.float32(high_action))
        high = np.ones(N_equity)*2.5
        low_bound = np.append(-high,0.)
        high_bound = np.append(high,self.T+1./365)
        self.observation_space = spaces.Box(low=np.float32(low_bound),high=np.float32(high_bound))


#current time start at zero
    def step(self, action): 
        assert self.action_space.contains(action)   
        if self.constraint == "only_long":
            action = action/np.sum(action)
        elif self.constraint == "long_short_limit":
            action = sign_renormalization(action,self.how_long,self.how_short)
        
        self.time_index = self.time_index + 1
        self.current_time = self.time_grid[self.time_index]
        self.current_asset = self.S_t[self.time_index-1]
        nu = self.sigma[self.time_index-1]*self.Identity@self.correlation_chole
        norm = np.linalg.norm(action@nu)
        drift = self.r_values[self.time_index-1] - (self.mu_values[self.time_index-1]@action)/norm
        self.I_t = self.I_t * (1 + self.dt * drift + np.sqrt(self.dt)*(self.target_vol/norm)*action@nu@self.W[self.time_index-1])
        
        if self.current_time < self.T:
            done = False
            reward = 0.
        else:
            done = True
            reward = np.maximum(self.I_t-self.strike_opt,0.)*self.discount
            self.simulation_index = self.simulation_index +1

        state = np.append(self.current_asset, self.current_time)
        return state, reward, done, {}


    def reset(self):
        if self.simulation_index == 0 or self.simulation_index==self.Nsim:
            self.simulations, self.Wiener, self.Vola = self.model.simulate(corr_chole = self.correlation_chole, random_gen = self.np_random, normalization = 1, Nsim=self.Nsim)
            self.simulation_index=0
        self.current_time = 0.
        self.time_index = 0
        self.I_t = self.I_0
        self.S_t = self.simulations[self.simulation_index]
        self.W = self.Wiener[self.simulation_index]
        self.sigma = self.Vola[self.simulation_index]
        state = np.append(np.zeros(self.N_equity), self.current_time)
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
