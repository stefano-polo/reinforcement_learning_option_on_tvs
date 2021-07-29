# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:32:18 2021

@author: u453878
"""


import numpy as np 
import statsmodels.api as sm
from scipy.interpolate import interp1d
import time
import matplotlib.pyplot as plt
import math

### personal libraries
from envs.pricing.pricing import LV_model, Vanilla_PayOff
from envs.pricing.loadfromtxt import LoadFromTxt
from envs.pricing.montecarlo import MC_Analisys


def nu_matrix(sigma_t_vector,identity_3d, corr_chole, N_equity, Nsim):
    S = sigma_t_vector.T*identity_3d
    n = S.T@corr_chole
    return n.transpose(1,2,0).reshape(N_equity,N_equity,Nsim)

def alpha_dot_nu(alpha_matrix,nu_matrix, N_equity, Nsim):
    a = np.sum((alpha_matrix.T*nu_matrix.transpose(1,0,2)),axis=1).T
    return a

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def evolve_TVS(index_start, I_t, dS_S, Vola, alpha, target_vola, instant_interest_rate, dt, N_euler_grid, N_equity, Nsim, identity_3d, corr_chole):
    s = 1.0
    I_t_p = np.ones(dS_S[:,0,0].shape)*I_t
    for t in range(index_start, N_euler_grid+index_start):
        nu = nu_matrix(Vola[:,t,:],identity_3d, corr_chole, N_equity, Nsim)
        omega = target_vola/np.linalg.norm(alpha_dot_nu(alpha, nu, N_equity, Nsim),axis=1)
        I_t_p = I_t_p * (1. + np.sum(alpha*dS_S[:,t,:],axis=1)*omega + dt * instant_interest_rate[t]*(1.-omega*s))
    return I_t_p


def plot_regression(states, V_realized, V_alpha, I_t, alpha, time, number_state):
    plt.scatter(states,V_realized, label="Realized")
    plt.scatter(states,V_alpha, c='red', label='Regression')
    plt.title(r"$I_t$ = "+str(round(I_t,4))+" --- alpha = "+str(round(alpha,4))+" --- time = "+str(round(time,3)))
    plt.xlabel("State "+str(number_state))
    plt.ylabel("$V_t$")
    plt.legend()
    plt.show()
    
def plot_regression3d(state1, state2,V_realized, V_alpha, I_t, alpha, time):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(state1, state2, V_realized, color="blue",label="Realized")
    ax.scatter(state1, state2, V_alpha, color="red",label="Regression")
    ax.set_xlabel("State 1")
    ax.set_ylabel("State 2")
    ax.set_zlabel("$V_t$")
    plt.title(r"$I_t$ = "+str(round(I_t,4))+" --- alpha = "+str(round(alpha,4))+" --- time = "+str(round(time,3)))
    plt.legend()
    plt.show()


def evolve_martingale(S_t, LV, random_gen, t_in, t_fin, cholesky, F, Nsim, N_equity):
    logX_t = np.zeros(S_t.shape) 
    S_t_plus_1 = np.zeros(S_t.shape)
    Z = random_gen.randn(Nsim,N_equity)
    ep = cholesky@Z.T
    dt = abs(t_fin - t_in)
    sigma = np.zeros(S_t.shape)
    for i in range(N_equity):
        logX_t[:,i] = np.log(S_t[:,i]/F[i](t_in))
        sigma[:,i] = LV[i](t_in, logX_t[:,i])
        logX_t[:,i] = logX_t[:,i]-0.5*dt*(sigma[:,i]**2)+sigma[:,i]*np.sqrt(dt)*ep[i]
        S_t_plus_1[:,i] = np.exp(logX_t[:,i])*F[i](t_fin)
    return S_t_plus_1, sigma


def compute_alphas(angle, Nsim, N_equity):
     alpha = np.zeros((Nsim, N_equity))   
     alpha[:,0] = angle
     alpha[:,1] = 1. - angle
     return alpha

Seed = 11
N_equity = 2
frequency = "month"
target_volatility = 5./100
I_0 = 1.
K = I_0     #option strike 
T = 1.      #option expiry
Nsim = 1e3 #number of MC paths
N_mc_forward = 2e5
print_logs=True
save_coeff = True
load_coeff = False
R = 1.
name = ""
number_I = 100
number_alpha= 2
polynomial_basis = lambda x, y:  np.column_stack((x, y, x**2, y**2, x*y))    #, x**3,y**3,y*x**2,x*y**2))


Nsim = int(Nsim)
N_mc_forward = int(N_mc_forward)
shape = (N_equity,N_equity,Nsim)
### Array of identity matrices for vectorized calculations
identity_3d = np.zeros(shape)
idx = np.arange(shape[0])
identity_3d[idx, idx, :] = 1

##Problem grids

#angles_grid = np.linspace(0.,np.pi*0.5,number_alpha)
angles_grid = np.linspace(0.,1.,number_alpha)       ## grid on the action 

ACT = 365.0
if frequency == "month":
    month = 30./365.
    t_grid = np.linspace(month,T,int(T*12))
    N_euler_grid = 60       
    state_index = np.arange(int(12*T)+1)*N_euler_grid
elif frequency == "day":
    t_grid = np.linspace(1./ACT, T, int(T*ACT))
    N_euler_grid = 2
    state_index = np.arange(int(365*T)+1)*N_euler_grid
t_grid = np.append(0.,t_grid)


##Reading Market Data and building model
names = ["DJ 50 EURO E","S&P 500 NET EUR"]#["DJ 50 EURO E","S&P 500 NET EUR"]
correlation = np.array(([1.,0.],[0.,1.]))
folder_name = "constant_vola_market"   #before_fake_smiles
D, F, V, LV = LoadFromTxt(names, folder_name)

mean_I = I_0*np.exp(D.R(T)*T)
var_I = (I_0**2)*np.exp(2.*D.R(T)*T)*(np.exp(T*target_volatility**2)-1.0)
n_sigma = 10.0
lower_bound = mean_I-n_sigma*np.sqrt(var_I)
upper_bound = mean_I+n_sigma*np.sqrt(var_I)
if(lower_bound<0):
    lower_bound = 0.0
I_grid = np.linspace(lower_bound,upper_bound,number_I)         

ATM_idx = find_nearest(I_grid, I_0)
spot_prices = np.array([])
for forward in F:
    spot_prices = np.append(spot_prices,forward.spot)
    
correlation_chole = np.linalg.cholesky(correlation)
model = LV_model(fixings=t_grid[1:], local_vol_curve=LV, forward_curve=F, N_grid = N_euler_grid)
euler_grid = model.time_grid
r_t = D.r_t(np.append(0.,euler_grid[:-1]))
dt_vector = model.dt
np_random = np.random
Polynomial_grade = len(polynomial_basis(1.,1.).T)
print("Polynomial grade ",Polynomial_grade)
coeff_matrices = np.zeros((len(I_grid),len(t_grid[:-1]),number_alpha, Polynomial_grade+1))
    
start_time = time.time()
##MC simulation of the risky assets 
np_random.seed(Seed)
S, simulations_Vola = model.simulate(corr_chole = correlation_chole, random_gen = np_random, normalization = 0, Nsim=Nsim)
S = np.insert(S,0,spot_prices,axis=1)
dS_S = (S[:,1:,:]-S[:,:-1,:])/S[:,:-1,:]


V_t_plus = np.zeros((Nsim, number_I))
state_index_rev = list(reversed(state_index[:-1]))
V_t = np.ones(V_t_plus.shape)*(-np.inf)
V_realized_matrix = np.zeros((Nsim,number_alpha))
V_I_save_matrix = np.zeros((Nsim, number_I, number_alpha, len(t_grid[:-1])))
k = 0
for I_T in I_grid:
    V_t_plus[:,k] = max(I_T-K,0.)
    k+=1

k=0
for t in reversed(t_grid[:-1]):
    start_index = state_index_rev[k]
    index_I_grid = 0
    state1 =S[:,start_index,0]/spot_prices[0]  #first asset 
    state2 = S[:,start_index,1]/spot_prices[1]
    Y = polynomial_basis(state1,state2)
    Y = np.insert(Y,0,1,axis=1)
    V_realized_interpolator = interp1d(I_grid, V_t_plus, axis=1, fill_value="extrapolate")
    for It in I_grid:
        angle_index = 0
        V_t[:,index_I_grid] = 0.
        I_t_plus_save = np.zeros((Nsim,number_alpha))
        for a in angles_grid:
            alpha = compute_alphas(a, Nsim, N_equity)
            I_t_plus_1 = evolve_TVS(start_index, It, dS_S, simulations_Vola, alpha, target_volatility, r_t, dt_vector[0], N_euler_grid, N_equity, Nsim, identity_3d, correlation_chole)
            V_realized = np.diag(V_realized_interpolator(I_t_plus_1))
            I_t_plus_save[:,angle_index] = V_realized
            ##Regression
            ols = sm.OLS(V_realized,Y)
            ols_result = ols.fit()
            coeff_It_t = ols_result.params
            coeff_matrices[index_I_grid, len(t_grid[:-1])-k-1, angle_index, :] = coeff_It_t
            ###Continuation value
            V_alpha = np.sum(coeff_It_t * Y, axis=1)     ##continuation value
            #V_realized_matrix[:,angle_index] = V_alpha 
            V_I_save_matrix[:,index_I_grid,angle_index,len(t_grid[:-1])-k-1] = V_realized
            V_t[:,index_I_grid] = np.maximum(V_t[:,index_I_grid],V_alpha)
            angle_index += 1 
        #V_t[:,index_I_grid] = np.max(V_realized_matrix,axis=1)   #metterlo fuori dal ciclo
        index_I_grid +=1
        
        if(((It == I_grid[0])  or (It==I_grid[ATM_idx]) or (It == I_grid[-1])) and (print_logs==True)):
            plot_regression3d(state1, state2,V_realized, V_alpha, It, a, t)
            plot_regression(state1, V_realized, V_alpha, It, a, t, "1")
            plot_regression(state2, V_realized, V_alpha, It, a, t, "2")
    
    k += 1
    V_t_plus = V_t
    V_t = np.ones(V_t_plus.shape)*(-np.inf)
    if print_logs:
        print(" Completed Month ",k," of ",T*12)
        print("Elapsed time ",(time.time()-start_time)/60., " min ")
  
    
if save_coeff:
    np.save(name+"Price_paths",V_t)
    np.save(name+"Regression_coeff",coeff_matrices)


if print_logs:
    t_LSMC = (time.time()-start_time)/60.
    print("--- Execution time for LSMC regression: ", t_LSMC, " min ---")
    print("Strating forward MC")

############    Forward Monte Carlo      ######################
 
t_MC = time.time()
### Array of identity matrices for vectorized calculations

shape = (N_equity,N_equity,N_mc_forward)
identity_3d = np.zeros(shape)
idx = np.arange(shape[0])
identity_3d[idx, idx, :] = 1
### MC arrays 
I_t = np.ones(N_mc_forward)*I_0
S_t = np.ones((N_mc_forward,N_equity))*spot_prices
alpha_best = np.array([])
nu = np.array([])
s = 1.0 #strategy sum

## Simulation
for i in range(len(t_grid[:-1])):
    ##variables for action choice
    V_max = np.ones(N_mc_forward)*(-np.inf)
    alpha_chosen = np.zeros((N_mc_forward,N_equity))    
    coeff_matrices_time =  coeff_matrices[:,i,:,:]      #select the regression coefficients for that time
    
    ##polynomial basis
    state1 = S_t[:,0]/spot_prices[0] #first asset 
    state2 = S_t[:,1]/spot_prices[1]
    Y = polynomial_basis(state1,state2)
    Y = np.insert(Y,0,1,axis=1)   ## add constant to the polynomial basis
    
    ## loop on the action grid to choose the best action 
    angle_index = 0
    for a in angles_grid:
        coeff_matrices_time_alpha = coeff_matrices_time[:,angle_index,:]   #select the regression coefficients for that action
        coefficients = interp1d(I_grid, coeff_matrices_time_alpha, axis=0, fill_value="extrapolate")(I_t) # find the regression coefficients for the simulated spot prices of the TVS
        expected_continuation = np.sum(coefficients * Y, axis=1)
        mask = expected_continuation > V_max
        alpha_chosen[mask,:] = np.array([a,1.0-a])#np.array([np.cos(a)**2,np.sin(a)**2])  #select the optimal action 
        V_max[mask] = expected_continuation[mask]
        angle_index += 1
    ## Market simulation: risky assets and TVS
    dt = (t_grid[i+1]-t_grid[i])/N_euler_grid
    t_in = t_grid[i] 
    for j in range(N_euler_grid):
        t_fin = t_in + dt
        dt = t_fin-t_in
        S_t_plus_1, sigma_t = evolve_martingale(S_t, LV, np_random, t_in, t_fin, correlation_chole, F, N_mc_forward, N_equity)
        nu = nu_matrix(sigma_t,identity_3d, correlation_chole, N_equity, N_mc_forward)
        if j==0:
            if print_logs:
                plt.hist(alpha_chosen[:,0], bins=50,  alpha=0.8,label=F[0].asset_name)
                plt.hist(alpha_chosen[:,1],bins=50,  alpha=0.8, label=F[1].asset_name)
                plt.title("Actions at time "+str(round(t_in,3)))
                plt.xlabel("Allocation Strategy")
                plt.ylabel("Frequency")
                plt.legend()
                plt.show()
            
        dS_S = (S_t_plus_1-S_t)/S_t
        omega = target_volatility/np.linalg.norm(alpha_dot_nu(alpha_chosen, nu, N_equity, N_mc_forward),axis=1)
        I_t = I_t*( 1. + np.sum(alpha_chosen*dS_S,axis=1)*omega + dt * D.r_t(t_in)*(1.-omega*s))
        S_t = S_t_plus_1
        t_in = t_fin
        
discounted_payoff = Vanilla_PayOff(I_t, K)*D(T)

x, mean, err = MC_Analisys(discounted_payoff, 100)
plt.errorbar(x, mean,yerr=err, label="LSMC")
plt.axhline(y=0.018160594508452997, label="Black optimal price",color="red")
#plt.axhline(y=0.007890500095994226,label="BS",color="red")
plt.xlabel("MC throws")
plt.show()
print("Call option price ", np.mean(discounted_payoff))
print("error", np.std(discounted_payoff)/np.sqrt(N_mc_forward))
plt.hist(I_t, bins=20)
plt.xlabel("I_t")
plt.ylabel("Frequencies")
plt.show()

if print_logs:
    t_MC = (time.time()-t_MC)/60.
    print("--- Execution time for forward MC: ", t_MC, " min ---")
    print("--- Total Execution time: ", t_MC + t_LSMC, " min ---")



state_index = [120]
k = 0
for t in t_grid[:-1]:
    for s in state_index:
        plt.plot(I_grid, V_I_save_matrix[s,:,0,k], label="alpha1 = [0, 1]")
        plt.plot(I_grid, V_I_save_matrix[s,:,-1,k], label="alpha2 = [1, 0]")
        plt.xlabel("I")
        plt.ylabel("V_t")
        plt.legend()
        plt.title("Time "+str(round(t,3)))
        plt.show()
        plt.plot(I_grid, V_I_save_matrix[s,:,0,k] -  V_I_save_matrix[s,:,-1,k], label="alpha1")
        plt.xlabel("I")
        plt.ylabel("V_t_alpha1 - V_t_alpha2")
        plt.title("Time "+str(round(t,3)))
        plt.show()
    k += 1