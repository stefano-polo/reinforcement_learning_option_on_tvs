from pricing.pricing import LV_model
from pricing.loadfromtxt import LoadFromTxt
from pricing.targetvol import optimization_only_long, CholeskyTDependent, Strategy,Drift,  Markowitz_solution
import numpy as np
import time
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

if MPI is None or MPI.COMM_WORLD.Get_rank()==0:
    rank = 0
else:
    rank = MPI.COMM_WORLD.Get_rank()

t0 = time.clock()
Nsim = int(1e5)
run = 0
Seed = run+rank#3000
strategy = "free"   #free
#print("Seme ",Seed)
I_0 = 1.
restart =0 
Black = 0
daily = 0
N_equity = 2                                #number of equities
target_vol = 5./100.
T = 2.
title = 'new/d'+str(int(T))+'/final_price_lvinvestigation_2_asset_equalrepo_long_Nsim_'+str(Nsim)+'_vola'+str(target_vol)+'_maturity'+str(T)+'_Black'+str(Black)+'_daily'+str(daily)+strategy
restart_file = 'day_new/d'+str(int(T-1))+'/final_price_lvinvestigation_2_asset_equalrepo_long_Nsim_8000_vola0.05_maturity'+str(T-1)+'_Black'+str(Black)+'_daily'+str(daily)+strategy

if restart:
    I_0 = np.loadtxt(restart_file+"_rank"+str(rank+run)+'.txt')
    Nsim =len(I_0)
else:
    I_0 = np.ones(Nsim)*I_0
"""Time grid creation for the simulation"""
if daily:
    days = 365
    n_days =days * int(T)
    observation_grid = np.linspace(1./days,T,n_days)
    observation_grid = np.insert(observation_grid,0,0.)
    time_index = 0
    current_time = 0.
    N_euler_grid = 2

else:
    month_dates = np.array([31.,28.,31.,30.,31.,30.,31.,31.,30.,31.,30.,31.])
    months = month_dates
    N_euler_grid = 60
    if T > 1.:
        for i in range(int(T)-1):
            months = np.append(months, month_dates)
    observation_grid = np.cumsum(months)/365.
    observation_grid = np.insert(observation_grid,0,0.)
 
simulation_index = 0
Identity = np.identity(N_equity)
"""Loading market curves"""
names = ["DJ 50 EURO E","S&P 500 NET EUR"]
D, F, V, LV = LoadFromTxt(names, "FakeSmilesDisplacedDiffusion")
correlation = np.identity(len(names))
spot_prices = np.ones(len(names))
for i in range(len(names)):
    spot_prices[i] = F[i].spot
   # print(F[i].q_values)
"""Preparing the LV model"""
#print(N_euler_grid)
model = LV_model(fixings=observation_grid[1:], local_vol_curve=LV, forward_curve=F, N_grid = N_euler_grid)
euler_grid = model.time_grid
discount = D(T)
print(discount)
dt_vector = model.dt
mu_function = Drift(forward_curves=F)
mean = np.array([])
correlation_chole = np.linalg.cholesky(correlation)
nu_function = CholeskyTDependent(variance_curves=V,correlation_chole= correlation_chole)
alpha = Strategy()
if strategy == "only_long":
    alpha.optimization_constrained(mu=mu_function,nu=nu_function,long_limit=25/100,N_trial=500,typo=1)
else: 
    alpha.Mark_strategy(mu=mu_function,nu=nu_function)
#final_price = np.zeros(Nsim)
generator = np.random
generator.seed(Seed)
S, simulations_Vola = model.simulate(corr_chole = correlation_chole, random_gen =generator, normalization = 0, Nsim=Nsim)
if not restart:
    S = np.insert(S,0,spot_prices,axis=1)
    r_t = D.r_t(np.append(0.,euler_grid[:-1]))
    mu_values = mu_function(np.append(0.,euler_grid[:-1]))
else:
    if daily:
        cut = int(365*(T-1))
    else:
        cut = int(12*(T-1))
    S = S[:,(cut*N_euler_grid-1):,:]
    simulations_Vola = simulations_Vola[:,cut*N_euler_grid:,:]
    observation_grid = observation_grid[cut:]
    euler_grid = model.time_grid[cut*N_euler_grid:]
    r_t = D.r_t(np.append(T,euler_grid[:-1]))
    mu_values = mu_function(np.append(T,euler_grid[:-1]))
dS_S = (S[:,1:,:] - S[:,:-1,:])/S[:,:-1,:]
#mean = np.array([])
for i in range(Nsim):
   # print(i)
    I_t = I_0[i]
   # counter = 0
   # counter_tot = 0
    sigma = simulations_Vola[i]
    norm_price = dS_S[i]
    for j  in range(len(observation_grid[1:])):
        dt = dt_vector[j]
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
                    if strategy=="only_long":
                        #action = optimization_only_long(mu, nu,seed=rank, guess = np.array(([0.4,0.6],[0.6,0.4])))
                        action = np.zeros(N_equity) 
                        action[np.argmax(sigma[idx])] = 1. 
                    else:
                        action =  Markowitz_solution(mu,nu,-1)
     #           action2 = alpha(observation_grid[j])
    #            f_base = (action@mu)/np.linalg.norm(action@nu)
      #          f_bl = (action2@mu)/np.linalg.norm(action2@nu)
       #         if f_bl>=f_base:
        #            counter = counter+1
         #       counter_tot = counter_tot+1
            prod = action@nu
            norm = np.sqrt(prod@prod)
            omega = target_vol/norm
            I_t = I_t * (1. + omega*action@norm_price[idx]  + (1 - omega*np.sum(action))*r_t[idx]*dt  )
    f = open(title+"_rank"+str(rank+run)+'.txt',"a")
    #final_price[i] = I_t
    f.write(str(I_t))
    f.write('\n')
    f.close()
  #  mean = np.append(mean,counter/counter_tot*100)
   # if i>0 and i%20==0:
    #    print("mean accuracy ",np.mean(mean))
    #simulations_Vola = np.delete(simulations_Vola,0,axis=0)
    #dS_S = np.delete(dS_S,0,axis=0)

print("EXECUTION TIME",(time.clock()-t0)/3600)
