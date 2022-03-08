import gym
from gym import spaces
from gym.utils import seeding
from numpy import array, exp


import sys
sys.path.insert(1, '../pricing')
from pricing import EquityForwardCurve, DiscountingCurve, ForwardVariance, Black
from closedforms import BS_European_option_closed_form

class PlainVanillaOption(gym.Env):
    """Plain Vanilla Option environment under single-asset Black and Scholes model"""
    def __init__(self, spot_price: float = 100.0, zero_interest_rate: float = 1/100, volatility: float = 20/100,
                 option_strike: float = 100.0, option_maturity: float = 1., call_put: int = 1,
                 n_sim_for_cache: int = int(1e4)) -> None:
        """
        Plain Vanilla Option environment under single-asset Black and Scholes model. The agent actions are dicrete: 0 not
        exercise, 1 exercise the right on the option.
        :param spot_price (float): initial spot price of the underlying asset
        :param zero_interest_rate (float): zero interest rate
        :param volatility (float): volatility of the underlying asset
        :param option_strike (float): strike price of the option
        :param option_maturity (float): maturity of the option
        :param call_put (int): 1 for call, -1 for put
        :param n_sim_for_cache (int, default=int(2e3)): number of simulated paths cached in memory
        (the cache speed-up the computation because it exploits the vectorization provided by numpy)
        """
        self.T = option_maturity
        self.strike_opt = option_strike
        self.asset_history = []
        self.cp = call_put
        self.Nsim = n_sim_for_cache
        self.simulation_index = 0
        t_max = 10.
        zero_interest_rate = array([zero_interest_rate, zero_interest_rate, zero_interest_rate])
        zero_interest_rate_dates = array([0.0, 5, t_max])
        d = exp(-zero_interest_rate*zero_interest_rate_dates)
        self.discounting_curve = DiscountingCurve(reference=0, discounts=d, discount_dates=zero_interest_rate_dates)
        self.forward_curve = EquityForwardCurve(reference=0, spot=spot_price, discounting_curve=self.discounting_curve,
                                                repo_dates=array([0., t_max]), repo_rates=array([0., 0.]))
        strike_spot_vola = array([spot_price, 200])
        spot_vol = array(([volatility, volatility], [volatility, volatility]))
        spot_vol_dates = array([0.1, t_max])
        self.variance_curve = ForwardVariance(reference=0, market_volatility_matrix=spot_vol, strikes=strike_spot_vola,
                                              maturity_dates=spot_vol_dates, strike_interp=spot_price)
        self.BS = Black(fixings=array([self.T]), forward_curve=self.forward_curve,
                        variance_curve=self.variance_curve, sampling="standard")
        # possible actions are exercise(1) or not (0)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=array([0., 0.]), high=array([spot_price*10000., self.T+1./365.]))
        self.seed()
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)

        if self.current_time < self.T:
            done = False
            self.current_asset = self.simulated_asset[self.simulation_index]
            self.simulation_index += 1
            self.current_time = self.T
            self.asset_history.append(self.current_asset)
            reward = 0
            return array((self.current_asset, self.current_time)), reward, done, {}

        if self.current_time == self.T:
            done = True
            if action:
                reward = self.cp * (self.current_asset - self.strike_opt)
            else:
                reward = 0
            return array((self.current_asset, self.current_time)), reward, done, {}

    def reset(self):
        if self.simulation_index == 0 or self.simulation_index == self.Nsim:
            self.simulated_asset = self.BS.simulate(random_generator=self.np_random, n_sim=self.Nsim)[:, 0]
            self.simulation_index = 0
        self.current_asset = self.forward_curve.spot
        self.current_time = 0.
        self.asset_history = []
        self.asset_history.append(self.forward_curve.spot)
        return array((self.current_asset, self.current_time))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        print()
        print('asset_history = ', self.asset_history)
        print('current time = ', self.current_time)

    def theoretical_price(self):
        return BS_European_option_closed_form(forward=self.forward_curve(self.T), strike=self.strike_opt,
                                              time_to_maturity=self.T, discount=self.discounting_curve(self.T),
                                              volatility=np.sqrt(self.variance_curve(self.T)), call_put=self.cp)
