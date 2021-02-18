from mpi4py import MPI
import numpy as np
from scipy.interpolate import interp1d
from time import time
from scipy.optimize import minimize
import xml.etree.ElementTree as ET
from pricing.read_market import MarketDataReader, Market_Local_volatility
from pricing.montecarlo import MC_Analisys, MC
from pricing.targetvol import Drift
from pricing.pricing import LocalVolatilityCurve, EquityForwardCurve, ForwardVariance, DiscountingCurve,piecewise_function,Vanilla_PayOff,PricingModel,LV_model
from pricing.targetvol import optimization_only_long,optimization_limit_position, CholeskyTDependent, Strategy


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
t_start = MPI.Wtime()
Black = 0
daily = 0
print('Starting simulation')
print('SONO NODO',rank)
I_0 = 1.
I_t = I_0
N_equity = 2                                #number of equities
T = 1.
target_vol = 5./100.
"""Time grid creation for the simulation"""
if daily:
    n_days =365 * int(T)
    observation_grid = np.linspace(1./n_days,T,n_days)
    observation_grid = np.insert(observation_grid,0,0.)
    time_index = 0
    current_time = 0.
    N_euler_grid = 2
    print("Daily adjustment of the stratey in the time grid: ",observation_grid)
    print("Number of discretization points for the Euler sampling of LV: ", N_euler_grid)
else:
    month_dates = np.array([31.,28.,31.,30.,31.,30.,31.,31.,30.,31.,30.,31.])
    months = month_dates
    N_euler_grid = 60
    if T > 1.:
       for i in range(int(T)-1):
            months = np.append(months, month_dates)
    observation_grid = np.cumsum(months)/365.
    observation_grid = np.insert(observation_grid,0,0.)
    print("Monthly adjustment of the stratey in the time grid: ",observation_grid)
    print("Number of discretization points for the Euler sampling of LV: ", N_euler_grid)  

simulation_index = 0
Nsim = int(2e3)
Identity = np.identity(N_equity)
Seed = rank
title = 'investigation_2assets_maturity1/final_price_lvinvestigation_2_asset_equalrepo_long_Nsim_'+str(Nsim)+'_vola'+str(target_vol)+'_maturity'+str(T)+'_Black'+str(Black)+'_daily'+str(daily)
"""Loading market curves"""
reader = MarketDataReader("TVS_example.xml")
D = reader.get_discounts()
spot_prices = reader.get_spot_prices()
F = reader.get_forward_curves()
F = [F[0],F[1],F[2],F[3],F[4],F[5],F[6],F[7],F[9]]
V = reader.get_volatilities()
V = [V[0],V[1],V[2],V[3],V[4],V[5],V[6],V[7],V[9]]
correlation = np.array(([1.,0.],[0.,1.]))
names = reader.get_stock_names()
names = [names[0],names[1],names[2],names[3],names[4],names[5],names[6],names[7],names[9]]
local_vol = Market_Local_volatility()
LV = [local_vol[0],local_vol[1],local_vol[2],local_vol[3],local_vol[6],local_vol[4],local_vol[5],local_vol[7],local_vol[8]]
F = []
spot_prices =  np.array([spot_prices[0],spot_prices[5]])
F.append(EquityForwardCurve(reference=0,discounting_curve=D, spot = spot_prices[0], repo_rates=-1.*np.array([0.0002,0.0001,0.00011]),repo_dates=np.array([30.,223.,466.])/365.))
F.append(EquityForwardCurve(reference=0,discounting_curve=D, spot = spot_prices[1], repo_rates=-1.*np.array([0.0001,0.0002,0.0001]),repo_dates=np.array([80.,201.,306.])/365.))
LV = [LV[0],LV[5]]
V = [V[0],V[5]]
names = [names[0],names[5]]
for i in range(N_equity):
    print(LV[i].name)
"""Preparing the LV model"""
model = LV_model(fixings=observation_grid[1:], local_vol_curve=LV, forward_curve=F, N_grid = N_euler_grid)
euler_grid = model.time_grid
r_t = D.r_t(np.append(0.,euler_grid[:-1]))
discount = D(T)
dt = model.dt[0]
mu_function = Drift(forward_curves=F)
mu_values = mu_function(np.append(0.,euler_grid[:-1]))
generator = np.random
generator.seed(Seed)
correlation_chole = np.linalg.cholesky(correlation)
if Black:
    print('Black Strategy for LV')
    nu_function = CholeskyTDependent(variance_curves=V,correlation_chole= correlation_chole)
    alpha = Strategy()
    alpha.optimization_constrained(mu=mu_function,nu=nu_function,long_limit=25/100,N_trial=500,typo=1)
final_price = np.zeros(Nsim)
S, simulations_Vola = model.simulate(corr_chole = correlation_chole, random_gen =generator, normalization = 0, Nsim=Nsim)
S = np.insert(S,0,spot_prices,axis=1)
dS_S = (S[:,1:,:] - S[:,:-1,:])/S[:,:-1,:]
if rank == 0:
    print('Finished simulating paths')
for i in range(Nsim):
    print(i)
    I_t = I_0
    sigma = simulations_Vola[0]
    norm_price = dS_S[0]
    for j  in range(len(observation_grid[1:])):
        index_plus = j*N_euler_grid
        for k in range(N_euler_grid):
            idx = index_plus + k 
            Vola =  sigma[idx]*Identity
            nu = Vola @ correlation_chole
            if k ==0:
                if Black:
                    action = alpha(observation_grid[j])
                else:
                    mu = mu_values[idx]
                    action = optimization_only_long(mu, nu,seed=rank)
            norm = np.linalg.norm(action@nu)
            omega = target_vol/norm
            I_t = I_t * (1. + omega*action@norm_price[idx]  + (1 - omega)*r_t[idx]*dt  )
    f = open(title+"_rank"+str(rank)+'.txt',"a")
    final_price[i] = I_t
    f.write(str(I_t))
    f.write('\n')
    f.close()
    simulations_Vola = np.delete(simulations_Vola,0,axis=0)
    dS_S = np.delete(dS_S,0,axis=0)
results = comm.gather(final_price,root=0)
if rank ==0:
    vector_results = results[0]
    for i in range(1,size):
        vector_results = np.append(vector_results,results[i])
    np.save(title+'_Seed'+str(Seed),vector_results)
    print("Saved")
t_finish = MPI.Wtime()
print('Wall time ',(t_finish-t_start)/60,' min')
