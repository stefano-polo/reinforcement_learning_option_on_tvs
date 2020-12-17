import gym
from gym import spaces
from gym.utils import seeding
from numpy import log, sqrt, exp
import numpy as np
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, LV_model, ForwardVariance, quad_piecewise
from envs.pricing.closedforms import European_option_closed_form
from envs.pricing.targetvol import Drift, Strategy, TVSForwardCurve, TargetVolatilityStrategy
from envs.pricing.fake_market_lv import load_fake_market_lv
from envs.pricing.read_market import MarketDataReader, Market_Local_volatility
from envs.pricing.n_sphere import sign_renormalization

class TVS_LV(gym.Env):
    """Target volatility strategy Option environment with a Local volatility model for the assets
    """
    def __init__(self, N_equity= 2, target_volatility=5/100, I_0 = 1., r=1/100., strike_opt=1., maturity=1., constraint = "only_long", action_bound=20./100., sum_long = None, sum_short=None):
        """Pricing parameters for the option"""
        self.constraint = constraint
        self.target_vol = target_volatility
        self.I_0 = I_0
        self.I_t = I_0
        self.strike_opt = strike_opt
        self.N_equity = N_equity                                #number of equities
        self.T = maturity
        """Time grid creation for the simulation"""
        self.Identity = np.identity(self.N_equity)
        months = np.array([31.,28.,31.,30.,31.,30.,31.,31.,30.,31.,30.,31.])
        self.observation_grid = np.cumsum(months)/365.
        self.observation_grid = np.insert(self.observation_grid,0,0.)
        self.time_index = 0
        self.current_time = 0.
        self.N_euler_grid = 60
        self.simulation_index = 0
        self.Nsim = 1e3
        """Loading market curves"""
        reader = MarketDataReader("TV_example.xml")
        D = reader.get_discounts()
        F = reader.get_forward_curves()
        F = [F[0],F[1],F[2],F[3],F[4],F[5],F[6],F[7],F[9]]
        correlation = reader.get_correlation()
        if self.N_equity == 3:
            correlation = np.array(([1.,0.86,0.],[0.86,1.,0.],[0.,0.,1.]))
        elif self.N_equity == 2:
            correlation = np.array(([1.,0.86],[0.86,1.]))
        else:   
            correlation = np.delete(correlation,-1, axis = 1)
            correlation = np.delete(correlation,-1, axis = 0)
        self.correlation_chole = np.linalg.cholesky(correlation)
        names = reader.get_stock_names()
        names = [names[0],names[1],names[2],names[3],names[4],names[5],names[6],names[7],names[9]]
        print("Names original",names)
        V = Market_Local_volatility()
        LV = [V[0],V[1],V[2],V[3],V[6],V[4],V[5],V[7],V[8]]
        if self.N_equity == 3:
            F = [F[0],F[3],F[4]]
            LV = [LV[0],LV[3],LV[4]]
        elif self.N_equity == 2:
            F = [F[0],F[3]]
            LV = [LV[0],LV[3]]
        print("Simulating equity: ")
        self.spot_prices = np.array([])
        for i in range(self.N_equity):
            print(LV[i].name)
        """Preparing the LV model"""
        self.model = LV_model(fixings=self.observation_grid[1:], local_vol_curve=LV, forward_curve=F, N_grid = self.N_euler_grid)
        self.euler_grid = self.model.time_grid
        self.r_t = D.r_t(np.append(0.,self.euler_grid[:-1]))
        self.discount = D(self.T)
        self.dt_vector = self.model.dt
        self.mu_values = Drift(forward_curves = F)(np.append(0.,self.euler_grid[:-1]))
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
        self.current_time = self.observation_grid[self.time_index]
        self.current_logX = self.logX_t[self.time_index-1]
        dt = self.dt_vector[self.time_index-1]
        index_plus = (self.time_index-1)*self.N_euler_grid
        """Simulation of I_t"""
        for i in range(self.N_euler_grid):
            idx = index_plus + i 
            Vola =  self.sigma_t[idx]*self.Identity
            nu = Vola@self.correlation_chole
            norm = np.linalg.norm(action@nu)
            omega = self.target_vol/norm
            drift = self.r_t[idx] - omega * (action@self.mu_values[idx])
            mart = action@Vola@self.W_corr_t[idx]
            self.I_t = self.I_t * (1. + drift*dt + np.sqrt(dt)*mart)
        
        if self.current_time < self.T:
            done = False
            reward = 0.
        else:
            done = True
            reward = np.maximum(self.I_t-self.strike_opt,0.)*self.discount
            self.simulation_index = self.simulation_index +1

        state = np.append(self.current_logX, self.current_time)
        return state, reward, done, {}


    def reset(self):
        if self.simulation_index == 0 or self.simulation_index==self.Nsim:
            self.simulations_logX, self.simulations_W_corr, self.simulations_Vola = self.model.simulate(corr_chole = self.correlation_chole, random_gen = self.np_random, normalization = 1, Nsim=self.Nsim)
            self.simulation_index=0
        self.current_time = 0.
        self.time_index = 0
        self.I_t = self.I_0
        self.logX_t = self.simulations_logX[self.simulation_index]
        self.W_corr_t = self.simulations_W_corr[self.simulation_index]
        self.sigma_t = self.simulations_Vola[self.simulation_index]
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
