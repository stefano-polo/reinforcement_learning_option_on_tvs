import gym
from gym import spaces
from gym.utils import seeding
from numpy import array, exp, linspace
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, ForwardVariance, Black
from envs.pricing.closedforms import European_option_closed_form

class PlainVanillaOption(gym.Env):
    """Plain Vanilla Option environment"""
    def __init__(self, spot = 100, drift = 1/100, volatility = 20/100, strike = 100, maturity = 1.):
        self.sigma = volatility
        self.r = drift
        self.T = maturity
        self.s0 = spot
        self.strike_opt = strike
        self.asset_history = []
        T_max = 10
        zero_interest_rate = array([self.r,self.r,self.r])
        zero_interest_rate_dates = array([0.0,5,T_max])
        d = exp(-zero_interest_rate*zero_interest_rate_dates)
        D = DiscountingCurve(reference = 0, discounts= d, dates=zero_interest_rate_dates)
        F = EquityForwardCurve(reference=0, discounting_curve=D, spot=self.s0,repo_dates=array([0.,T_max]), repo_rates=array([0.,0.]))
        K_spot_vola = array([self.s0,200])
        spot_vol = array(([self.sigma,self.sigma],[0.3,0.3]))
        spot_vol_dates = array([0.1,T_max])
        V = ForwardVariance(reference=0,maturities=spot_vol_dates,strikes=K_spot_vola,spot_volatility=spot_vol,strike_interp=self.s0)
        self.BS = Black(variance=V, forward_curve=F)
        # possible actions are exercise(1) or not (0)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=array([0.,0.]),high=array([self.s0*10000.,self.T+1./365.]))
        self.seed()
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)

        if self.current_time < self.T:
            done = False
            self.current_asset = self.BS.simulate(fixings=array([self.T]), random_gen = self.np_random)[0,0]
            self.current_time = self.T
            self.asset_history.append(self.current_asset)
            reward = 0
            return array((self.current_asset,self.current_time)), reward, done, {}

        if self.current_time == self.T:
            done = True
            if action:
                reward = (self.current_asset - self.strike_opt)*exp(-self.r*self.T)
            else:
                reward = 0
            return array((self.current_asset,self.current_time)), reward, done, {}

    def reset(self):
        self.current_asset = self.s0
        self.current_time = 0.
        self.asset_history = []
        self.asset_history.append(self.s0)
        return array((self.current_asset, self.current_time))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        print()
        print('asset_history = ', self.asset_history)
        print('current time = ', self.current_time)

    def theoretical_price(self):
        return European_option_closed_form(forward = self.s0*exp(self.r*self.T), strike= self.strike_opt, maturity=self.T, reference=0, zero_interest_rate = self.r, volatility=self.sigma, typo = 1)
