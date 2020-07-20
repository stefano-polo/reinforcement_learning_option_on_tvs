import gym
from gym import spaces
from gym.utils import seeding
from numpy import sqrt
from numpy import exp
from numpy import array
from numpy import log
from numpy import linspace, zeros, append, ones
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, Black, ForwardVariance
from envs.pricing.closedforms import European_option_closed_form
from envs.pricing.targetvol import Drift, CholeskyTDependent, Strategy, TVSForwardCurve, TargetVolatilityStrategy
from envs.pricing.read_market import MarketDataReader


class TVS_simple(gym.Env):
    """Compound Option environment"""
    def __init__(self, N_equity= 2, target_volatility=0.2, r=1/100., strike_opt=100., maturity=1.):
        self.reference =0
        self.r = r
        self.spot_prices = array([100,200])
        self.N_equity = 2                                #number of equities
        T_max = 10
        self.correlation = array(([1,0.5],[0.5,1]))       #correlation matrix
        r_t = array([r,r,r,r])
        T_discounts = array([0.,3,6.,T_max])      #data observation of the market discounts factor
        market_discounts = exp(-r_t*T_discounts)       #market discounts factor
        T_repo1 = array([2,6,T_max])       #data observation of the market repo rates for equity 1
        repo_rate1 = array([0.22,0.22,0.22])/100  #market repo rates for equity 1
        T_repo2 = array([3.,6.,T_max])
        repo_rate2 = array([0.22,0.22,0.22])/100
        sigma1 = array([30,30.,30.])/100
        T_sigma1 = array([2,5.,T_max])
        K1 = array([self.spot_prices[0],500])
        spot_vola1 = array((sigma1,sigma1))                                      #market implied volatility matrix
        sigma2 = array([20,20,20])/100
        T_sigma2 =  array([2.,6,T_max])
        K2 = array([spot_price[1],600])
        spot_vola2 = array((sigma2,sigma2))
        self.D = DiscountingCurve(reference=self.reference, discounts=market_discounts,dates=T_discounts)
        self.F = np.array([])
        self.V = np.array([])
        for i in range(self.N_equity):
            q = globals()["repo_rate" + str(i+1)]
            T_q = globals()["T_repo" + str(i+1)]
            s_vola = globals()["spot_vola" + str(i+1)]
            T_vola = globals()["T_sigma" + str(i+1)]
            K = globals()["K" + str(i+1)]
            self.F = np.append(self.F,EquityForwardCurve(reference=t,spot=self.spot_prices[i],discounting_curve=D,repo_dates=T_q,repo_rates=q))
            self.V = np.append(self.V,ForwardVariance(reference=t,spot_volatility=s_vola,strikes=K,maturities=T_vola,strike_interp=self.spot_prices[i]))
        self.model = Black(variance=self.V,forward_curve = self.F)
        self.mu = Drift(self.F)
        self.nu = CholeskyTDependent(self.V,corr)
        self.T = maturity
        n_days = self.T*365
        self.time_grid = linspace(0,self.T,n_days)
        self.time_index = 0
        self.asset_history = []
        self.alpha_t = array([])
        # possible actions are exercise(1) or not (0)
        low_action = zeros(self.N_equity)
        high_action = ones(self.N_equity)
        self.action_space = spaces.Box(low = low_action, high = high_action)            #due azioni binarie
        low_bound = zeros(N_equity+1)
        high_bound = append(self.spot_prices*10000,self.T+1/365)
        self.observation_space = spaces.Box(low=low_bound,high=high_bound)   #è un continuo, dove si tratta di un box per il prezzo e il tempo

        self.seed()
        self.reset()

#current time start at zero
    def step(self, action):  # metodo che mi dice come evolve il sistema una volta arrivata una certa azione
        assert self.action_space.contains(action)   #gli arriva una certa azione

        if self.current_time == 0:
            self.S_t = self.model.simulate(fixings=self.time_grid, Ndim= self.N_equity, corr = self.correlation, random_gen = self.np_random)[0]

        if self.current_time < self.T:
            self.current_time = self.time_grid[self.time_index]
            self.current_asset = self.S_t[self.time_index]
            self.alpha_t = append(self.alpha_t, action)
            self.time_index = self.time_index+1
            state = append(self.current_asset, self.current_time)
            done = False
            reward = 0.
            return state, reward, done, {}

      ##### CASE t1: decide whether to enter the option or not #####
        if self.current_time == self.T:
            done = True
            TVSF = TVSForwardCurve(reference=self.reference)
            reward = - self.strike_opt + self.current_asset
            return array((self.current_asset,self.current_time)), reward, done, {}


    def reset(self):
        self.current_asset = self.spot_prices
        self.current_time = 0.
        self.alpha_t = array([])
        self.time_index = 0
        self.asset_history = []
        self.asset_history.append(self.spot_prices)
        return array((self.current_asset, self.current_time))


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def render(self, mode='human'):
        print()
        print('asset_history = ', self.asset_history)
        print('current time = ', self.current_time)


    def theoretical_price(self):
        return European_option_closed_form(forward = self.s0*exp(self.r*self.t1), strike= self.strike_opt, maturity=self.t1, reference=0, zero_interest_rate = self.r, volatility=self.sigma, typo=1)
