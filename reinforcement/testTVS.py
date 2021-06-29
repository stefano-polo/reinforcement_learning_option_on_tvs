# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 17:28:46 2021

@author: u453878
"""


import numpy as np 
import statsmodels.api as sm
from scipy.interpolate import interp1d
import time
import matplotlib.pyplot as plt

### personal libraries
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, LV_model, ForwardVariance, quad_piecewise, Vanilla_PayOff
from envs.pricing.targetvol import Drift
from envs.pricing.loadfromtxt import LoadFromTxt
from envs.pricing.targetvol import Markowitz_solution
from envs.pricing.closedforms import European_option_closed_form
from envs.pricing.montecarlo import MC_Analisys

def nu_matrix(sigma_t_vector,identity_3d, corr_chole, N_equity, Nsim):
    S = sigma_t_vector.T*identity_3d
    n = S.T@corr_chole
    return n.transpose(1,2,0).reshape(N_equity,N_equity,Nsim)

def alpha_dot_nu(alpha_matrix,nu_matrix, N_equity, Nsim):
    a = np.sum((alpha_matrix.T*nu_matrix.transpose(1,0,2)),axis=1).T
    return a


def evolve_martingale(S_t, LV, random_gen, t_in, t_fin, cholesky, F, Nsim, N_equity):
    logX_t = np.zeros(S_t.shape) 
    S_t_plus_1 = np.zeros(S_t.shape)
    Z = random_gen.randn(Nsim,N_equity)
    ep = cholesky@Z.T
    dt = abs(t_fin - t_in)
   # print("Dt in the martingale ",dt)
    sigma = np.zeros(S_t.shape)
    for i in range(N_equity):
        logX_t[:,i] = np.log(S_t[:,i]/F[i](t_in))
        sigma[:,i] = LV[i](t_in, logX_t[:,i])
        logX_t[:,i] = logX_t[:,i]-0.5*dt*(sigma[:,i]**2)+sigma[:,i]*np.sqrt(dt)*ep[i]
        S_t_plus_1[:,i] = np.exp(logX_t[:,i])*F[i](t_fin)
    return S_t_plus_1, sigma

Seed = 13
N_equity = 2
frequency = "month"
target_volatility = 5./100
I_0 = 1.
K = I_0     #option strike 
T = 1.0      #option expiry
N_mc_forward = 1e6
print_logs = True
N_euler_grid = 70
names = ["S&P 500 NET EUR","DJ 50 EURO E"]
N_equity = len(names)
correlation = np.array(([1.,0.],[0.,1.]))
folder_name = "constant_vola_market"   #before_fake_smiles
D, F, V, LV = LoadFromTxt(names, folder_name)
spot_prices = np.array([])

for forward in F:
    spot_prices = np.append(spot_prices,forward.spot)
correlation_chole = np.linalg.cholesky(correlation)
N_mc_forward = int(N_mc_forward)
I_t = np.ones(N_mc_forward)*I_0
S_t = np.ones((N_mc_forward,N_equity))*spot_prices
alpha_best = np.array([])
nu = np.array([])
#t_grid = np.linspace(T/12., T, 12 )
t_grid = np.array([31./365,62./365.])
dt = np.diff(t_grid)[0]
t_grid = np.append(0.,t_grid)
random_gen = np.random
random_gen.seed(Seed)
shape = (N_equity,N_equity,N_mc_forward)
### Array of identity matrices for vectorized calculations
identity_3d = np.zeros(shape)
idx = np.arange(shape[0])
identity_3d[idx, idx, :] = 1
action = np.array([1.0,0.0])
action_vector = np.ones((N_mc_forward, N_equity))*action 
s = 1.0

for i in range(len(t_grid[:-1])):
    t_in = t_grid[i] 
    dt = (t_grid[i+1]-t_grid[i])/N_euler_grid
    if print_logs:
        plt.hist(I_t, bins=50)
        plt.title("time "+str(t_in))
        plt.show()
    for j in range(N_euler_grid):
        t_fin = t_in + dt
        S_t_plus_1, sigma_t = evolve_martingale(S_t, LV, random_gen, t_in, t_fin, correlation_chole, F, N_mc_forward, N_equity)
        nu = nu_matrix(sigma_t,identity_3d, correlation_chole, N_equity, N_mc_forward)
        dS_S = (S_t_plus_1-S_t)/S_t
        omega = target_volatility/np.linalg.norm(alpha_dot_nu(action_vector, nu, N_equity, N_mc_forward),axis=1)
        I_t = I_t*( 1.0 + np.sum(action_vector*dS_S,axis=1)*omega + dt * D.r_t(t_in)*(1.-omega*s))
        S_t = S_t_plus_1
        t_in = t_fin
        
discounted_payoff = Vanilla_PayOff(I_t, K)*D(T)

x, mean, err = MC_Analisys(discounted_payoff, 100)
plt.errorbar(x, mean,yerr=err, label="LSMC")
#plt.axhline(y=0.017645356188118348, label="Black optimal price",color="red")
plt.axhline(y=0.007890500095994226,label="BS",color="red")
plt.xlabel("MC throws")
plt.show()
print("Call option price ", np.mean(discounted_payoff))
print("error", np.std(discounted_payoff)/np.sqrt(N_mc_forward))
plt.hist(I_t, bins=20)
plt.xlabel("I_t")
plt.ylabel("Frequencies")