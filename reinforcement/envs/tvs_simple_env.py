import gym
from gym import spaces
from gym.utils import seeding
from numpy import sqrt
from numpy import exp
from numpy import array
from numpy import log
from numpy import linspace, zeros, append, ones, vstack, maximum
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, Black, ForwardVariance
from envs.pricing.closedforms import European_option_closed_form
from envs.pricing.targetvol import Drift, CholeskyTDependent, Strategy, TVSForwardCurve, TargetVolatilityStrategy
from envs.pricing.read_market import MarketDataReader


class TVS_simple(gym.Env):
    """Compound Option environment"""
    def __init__(self, N_equity= 2, target_volatility=0.1, I_0 = 100, r=1/100., strike_opt=100., maturity=1./12):
        self.reference =0
        self.r = r
        self.target_vol = target_volatility
        self.spot_I = I_0
        self.spot_prices = array([100,200])
        self.N_equity = N_equity                                #number of equities
        self.asset_history = array([])
        self.correlation = array(([1,0.5],[0.5,1]))       #correlation matrix
        self.T = maturity
        n_days = 30
        self.time_grid = linspace(0,self.T,n_days)
        self.time_index = 1
        self.asset_history = array([])
        self.alpha_t = array([])
        self.strike_opt = strike_opt

        T_max = 10
        r_t = array([r,r,r,r])
        T_discounts = array([0.,3,6.,T_max])      #data observation of the market discounts factor
        market_discounts = exp(-r_t*T_discounts)       #market discounts factor
        T_repo1 = array([2,6,T_max])       #data observation of the market repo rates for equity 1
        repo_rate1 = array([0.32,0.32,0.32])/10  #market repo rates for equity 1
        T_repo2 = array([3.,6.,T_max])
        repo_rate2 = array([0.22,0.22,0.22])/10
        sigma1 = array([20,20.,20.])/100
        T_sigma1 = array([2,5.,T_max])
        K1 = array([self.spot_prices[0],500])
        spot_vola1 = array((sigma1,sigma1))                                      #market implied volatility matrix
        sigma2 = array([20,20,20])/100
        T_sigma2 =  array([2.,6,T_max])
        K2 = array([self.spot_prices[1],600])
        spot_vola2 = array((sigma2,sigma2))
        self.D = DiscountingCurve(reference=self.reference, discounts=market_discounts,dates=T_discounts)
        self.F = array([])
        self.V = array([])
        q = repo_rate1
        T_q = T_repo1
        s_vola = spot_vola1
        T_vola = T_sigma1
        K = K1
        self.F = append(self.F,EquityForwardCurve(reference=self.reference,spot=self.spot_prices[0],discounting_curve=self.D,repo_dates=T_q,repo_rates=q))
        self.V = append(self.V,ForwardVariance(reference=self.reference,spot_volatility=s_vola,strikes=K,maturities=T_vola,strike_interp=self.spot_prices[0]))
        q = repo_rate2
        T_q = T_repo2
        s_vola = spot_vola2
        T_vola = T_sigma2
        K = K2
        self.F = append(self.F,EquityForwardCurve(reference=self.reference,spot=self.spot_prices[1],discounting_curve=self.D,repo_dates=T_q,repo_rates=q))
        self.V = append(self.V,ForwardVariance(reference=self.reference,spot_volatility=s_vola,strikes=K,maturities=T_vola,strike_interp=self.spot_prices[1]))
        self.model = Black(variance=self.V,forward_curve = self.F)
        self.mu = Drift(self.F)
        self.nu = CholeskyTDependent(self.V,self.correlation)

        # possible actions are exercise(1) or not (0)
        low_action = ones(N_equity)*1e-6
        high_action = ones(self.N_equity)
        self.action_space = spaces.Box(low = low_action, high = high_action)            #due azioni binarie
        low_bound = zeros(1+N_equity)
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
            if self.time_index == 1:
                self.alpha_t = action
            else:
                self.alpha_t = vstack([self.alpha_t, action])
            state = append(self.current_asset, self.current_time)
            self.asset_history = append(self.asset_history,self.current_asset)
            done = False
            reward = 0.
            self.time_index = self.time_index+1
            return state, reward, done, {}

        if self.current_time == self.T:
            done = True
            self.alpha_t = vstack([self.alpha_t, action])
            self.current_asset = self.S_t[self.time_index-1]
            self.current_time = self.time_grid[self.time_index-1]
            state = append(self.current_asset, self.current_time)
            self.asset_history = append(self.asset_history,self.current_asset)
            alpha = Strategy(strategy = self.alpha_t, dates = self.time_grid)
            TVSF = TVSForwardCurve(reference = self.reference, vola_target = self.target_vol, spot_price = self.spot_I, strategy = alpha, mu = self.mu, nu = self.nu, discounting_curve = self.D)
            TVS = TargetVolatilityStrategy(forward_curve=TVSF)
            I_t = TVS.simulate(fixings=array([self.T]), random_gen=self.np_random)[0,0]
            reward = maximum(I_t-self.strike_opt,0)*self.D(self.T)
            return state, reward, done, {}


    def reset(self):
        self.current_asset = self.spot_prices
        self.current_time = 0.
        self.alpha_t = array([])
        self.time_index = 1
        self.asset_history = array([])
        self.asset_history = append(self.asset_history,self.spot_prices)
        state = append(self.current_asset, self.current_time)
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
