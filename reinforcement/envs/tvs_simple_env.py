import gym
from gym import spaces
from gym.utils import seeding
from numpy import log, sqrt, exp
import numpy as np
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, Black, ForwardVariance
from envs.pricing.closedforms import European_option_closed_form
from envs.pricing.targetvol import Drift, CholeskyTDependent, Strategy, TVSForwardCurve, TargetVolatilityStrategy
from envs.pricing.read_market import MarketDataReader
from envs.pricing.n_sphere import n_sphere_to_cartesian

class TVS_simple(gym.Env):
    """Target volatility strategy Option environment with a simple market"""
    def __init__(self, N_equity= 2, target_volatility=5/100, I_0 = 1., r=1/100., strike_opt=1., maturity=1., constraint = "only_long", action_bound=50, sum_long = None, sum_short=None):
        self.reference =0
        self.r = r
        self.constraint = constraint
        self.target_vol = target_volatility
        self.spot_I = I_0
        self.N_equity = N_equity                                #number of equities
        self.asset_history = np.array([])
        self.T = maturity
        n_days = 12
        self.time_grid = np.linspace(0,self.T,n_days)
        self.time_index = 0
        self.asset_history = np.array([])
        self.alpha_t = np.array([])
        self.strike_opt = strike_opt
        if N_equity ==3:
            self.correlation = np.array(([1,0.5,0.3],[0.5,1,0.3],[0.3,0.3,1]))       #correlation matrix
            self.spot_prices = np.array([100,200,89])
            T_repo3 = np.array([2/12.,5/12.,maturity])
            repo_rate3 = np.array([0.02,0.02,0.02])/10
        else:
            self.correlation = np.array(([1,0.5],[0.5,1]))       #correlation matrix
            self.spot_prices = np.array([100,200])

        r_t = np.array([r,r,r,r])
        T_discounts = np.array([0.,3/12,4/12.,maturity])      #data observation of the market discounts factor
        market_discounts = exp(-r_t*T_discounts)       #market discounts factor
        T_repo1 = np.array([1/12,4./12,maturity])       #data observation of the market repo rates for equity 1
        repo_rate2 = np.array([0.72,0.42,0.02])/10  #market repo rates for equity 1 0.52
        T_repo2 = np.array([1/12.,4/12.,maturity])
        repo_rate1 = np.array([0.22,0.22,0.22])/10
        
        sigma1 = np.array([20,20.,20.])/100
        T_sigma1 = np.array([2/12,5./12,maturity])
        K1 = np.array([self.spot_prices[0],500])
        spot_vola1 = np.array((sigma1,sigma1))                                      #market implied volatility matrix
        sigma2 = np.array([20,20,20])/100
        T_sigma2 =  np.array([2/12.,6/12,maturity])
        K2 = np.array([self.spot_prices[1],600])
        spot_vola2 = np.array((sigma2,sigma2))
        if self.N_equity==3:
            sigma3 = np.array([10,10,10])/100
            T_sigma3 =  np.array([2/12.,6/12,maturity])
            K3 = np.array([self.spot_prices[2],600])
            spot_vola3 = np.array((sigma3,sigma3))

        self.D = DiscountingCurve(reference=self.reference, discounts=market_discounts,dates=T_discounts)
        self.F = []
        self.V = []
        q = repo_rate1
        T_q = T_repo1
        s_vola = spot_vola1
        T_vola = T_sigma1
        K = K1
        self.F.append(EquityForwardCurve(reference=self.reference,spot=self.spot_prices[0],discounting_curve=self.D,repo_dates=T_q,repo_rates=q))
        self.V.append(ForwardVariance(reference=self.reference,spot_volatility=s_vola,strikes=K,maturities=T_vola,strike_interp=self.spot_prices[0]))
        q = repo_rate2
        T_q = T_repo2
        s_vola = spot_vola2
        T_vola = T_sigma2
        K = K2
        self.F.append(EquityForwardCurve(reference=self.reference,spot=self.spot_prices[1],discounting_curve=self.D,repo_dates=T_q,repo_rates=q))
        self.V.append(ForwardVariance(reference=self.reference,spot_volatility=s_vola,strikes=K,maturities=T_vola,strike_interp=self.spot_prices[1]))
        if self.N_equity==3:

            q = repo_rate3
            T_q = T_repo3
            s_vola = spot_vola3
            T_vola = T_sigma3
            K = K3
            self.F.append(EquityForwardCurve(reference=self.reference,spot=self.spot_prices[2],discounting_curve=self.D,repo_dates=T_q,repo_rates=q))
            self.V.append(ForwardVariance(reference=self.reference,spot_volatility=s_vola,strikes=K,maturities=T_vola,strike_interp=self.spot_prices[2]))
        
        self.model = Black(variance=self.V,forward_curve = self.F)
        self.mu = Drift(self.F)
        self.nu = CholeskyTDependent(self.V,self.correlation)

        if self.constraint == 'long_short_limit' and (sum_long is None or sum_short is None):
            raise Exception("You should provide the sum limit for short and long position")
        if sum_long is not None and sum_short is not None:
            self.sum_long = sum_long
            self.sum_short = sum_short
        if self.constraint != "only_long":
            low_action = np.ones(self.N_equity)*(-abs(action_bound))
            high_action = np.ones(self.N_equity)*abs(action_bound)
        else:
            if self.N_equity > 2:
                low_action = np.zeros(self.N_equity-1)
                high_action = np.ones(self.N_equity-2)*np.pi
                high_action = np.append(high_action, np.pi*2)
            else:
                low_action = np.zeros(self.N_equity-1)
                high_action = np.ones(1)*np.pi*2
        self.action_space = spaces.Box(low = np.float32(low_action), high = np.float32(high_action))
        high = log(np.ones(N_equity)*1000)
        low_bound = np.append(-high,0)
        high_bound = np.append(high,self.T+1/365)
        self.observation_space = spaces.Box(low=np.float32(low_bound),high=np.float32(high_bound))   
        self.seed()
        self.reset()

#current time start at zero
    def step(self, action):  # metodo che mi dice come evolve il sistema una volta arrivata una certa azione
        assert self.action_space.contains(action)   #gli arriva una certa azione
        if self.constraint == "only_long":
            action = n_sphere_to_cartesian(1,action)**2
        elif self.constraint == "long_short_limit":
            action = sign_renormalization(action,self.how_long,self.how_short)

        if self.time_index == 0:
            self.S_t = log(self.model.simulate(fixings=self.time_grid, corr = self.correlation, random_gen = self.np_random)[0]/self.spot_prices)
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
            alpha = Strategy(strategy = self.alpha_t, dates = self.time_grid[1:])
            TVSF = TVSForwardCurve(reference = self.reference, vola_target = self.target_vol, spot_price = self.spot_I, strategy = alpha, mu = self.mu, nu = self.nu, discounting_curve = self.D)
            TVS = TargetVolatilityStrategy(forward_curve=TVSF)
            I_t = TVS.simulate(fixings=np.array([self.T]), random_gen=self.np_random)[0,0]
            reward = np.maximum(I_t-self.strike_opt,0)*self.D(self.T)
        
        self.asset_history = np.append(self.asset_history,self.current_asset)
        state = np.append(self.current_asset, self.current_time)
        return state, reward, done, {}


    def reset(self):
        self.current_asset = np.zeros(len(self.F))
        self.current_time = 0.
        self.alpha_t = np.array([])
        self.time_index = 0
        self.asset_history = np.array([])
        self.asset_history = np.append(self.asset_history,self.current_asset)
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
        s_right.Mark_strategy(mu = self.mu, nu = self.nu)
        TVSF = TVSForwardCurve(reference = self.reference, vola_target = self.target_vol, spot_price = self.spot_I, strategy = s_right, mu = self.mu, nu = self.nu, discounting_curve = self.D)
        return European_option_closed_form(TVSF(self.T),self.strike_option,self.T,0,self.r,self.target_vol,1)
