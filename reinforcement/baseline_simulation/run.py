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
from pricing.targetvol import optimization_only_long,optimization_limit_position


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
t_start = MPI.Wtime()
print('Starting simulation')
I_0 = 1.
I_t = I_0
N_equity = 3                                #number of equities
T = 1.
target_vol = 5./100.
"""Time grid creation for the simulation"""
Identity = np.identity(N_equity)
n_days =365 
observation_grid = np.linspace(T/n_days,T,n_days)
observation_grid = np.insert(observation_grid,0,0.)
time_index = 0
current_time = 0.
N_euler_grid = 3
simulation_index = 0
Nsim = int(2e3)#int(15625)
Seed = rank+4*rank
"""Loading market curves"""
reader = MarketDataReader("TVS_example.xml")
D = reader.get_discounts()
F = reader.get_forward_curves()
F = [F[0],F[1],F[2],F[3],F[4],F[5],F[6],F[7],F[9]]
correlation = np.array(([1.,0.86,0.],[0.86,1.,0.],[0.,0.,1.]))
correlation_chole = np.linalg.cholesky(correlation)
names = reader.get_stock_names()
names = [names[0],names[1],names[2],names[3],names[4],names[5],names[6],names[7],names[9]]
local_vol = Market_Local_volatility()
LV = [local_vol[0],local_vol[1],local_vol[2],local_vol[3],local_vol[6],local_vol[4],local_vol[5],local_vol[7],local_vol[8]]

#F = []
#F.append(EquityForwardCurve(reference=0,discounting_curve=D, spot = 100., repo_rates=-1.*np.array([0.00012,0.0001,0.0001]),repo_dates=np.array([30.,123.,466.])/365.))
#F.append(EquityForwardCurve(reference=0,discounting_curve=D, spot = 100., repo_rates=-1.*np.array([0.0001,0.0001,0.0001]),repo_dates=np.array([80.,201.,306.])/365.))
#F.append(EquityForwardCurve(reference=0,discounting_curve=D, spot = 100., repo_rates=np.array([0.0001,0.0001,0.0001]),repo_dates=np.array([0.1,2.,10.])))
F = [F[0],F[3],F[4]]
LV = [LV[0],LV[3],LV[4]]
names = [names[0],names[3],names[4]]
for i in range(N_equity):
    print(LV[i].name)
"""Preparing the LV model"""
model = LV_model(fixings=observation_grid[1:], local_vol_curve=LV, forward_curve=F, N_grid = N_euler_grid)
euler_grid = model.time_grid
r_t = D.r_t(np.append(0.,euler_grid[:-1]))
discount = D(T)
dt = model.dt[0]
mu_values = Drift(forward_curves = F)(np.append(0.,euler_grid[:-1]))
generator = np.random
generator.seed(Seed)
final_price = np.zeros(Nsim)

simulations_W_corr, simulations_Vola = model.simulate(corr_chole = correlation_chole, random_gen =generator, normalization = 1, Nsim=Nsim)
if rank == 0:
    print('Finished simulating paths')
for i in range(Nsim):
    I_t = I_0
    sigma = simulations_Vola[0]
    wiener = simulations_W_corr[0]
    for j  in range(len(observation_grid[1:])):
        index_plus = j*N_euler_grid
        for k in range(N_euler_grid):
            idx = index_plus + k 
            Vola =  sigma[idx]*Identity
            nu = Vola@correlation_chole
            mu = mu_values[idx]
            if j ==0:
                action = optimization_only_long(mu, nu,seed=rank)
            norm = np.linalg.norm(action@nu)
            omega = target_vol/norm
            drift = r_t[idx] - omega * (action@mu)
            mart = action@Vola@wiener[idx]
            I_t = I_t * (1. + drift*dt + np.sqrt(dt)*omega*mart)
    final_price[i] = I_t
    simulations_Vola = np.delete(simulations_Vola,0,axis=0)
    simulations_W_corr = np.delete(simulations_W_corr,0,axis=0)
results = comm.gather(final_price,root=0)
if rank ==0:
    vector_results = results[0]
    for i in range(1,size):
        vector_results = np.append(vector_results,results[i])
    np.save('investigation_3assets/final_price_lvinvestigation_3_asset_long_Nsim_'+str(Nsim)+'_Seed'+str(Seed),vector_results)
    print("Saved")
t_finish = MPI.Wtime()
print('Wall time ',(t_finish-t_start)/60,' min')
