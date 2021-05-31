import numpy as np 
import statsmodels.api as sm
from scipy.interpolate import interp1d
import time
import matplotlib.pyplot as plt


### personal libraries
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, LV_model, ForwardVariance, quad_piecewise
from envs.pricing.targetvol import Drift
from envs.pricing.loadfromtxt import LoadFromTxt
from envs.pricing.targetvol import Markowitz_solution
from envs.pricing.closedforms import European_option_closed_form


def alpha_vector(a1, nu, Nsim=1, norm=1):
    a_1 = np.ones(Nsim)*a1
    beta = nu[1,0,:]**2+nu[1,1,:]**2
    delta = (a_1**2)*(nu[0,0,:]**2+nu[0,1,:]**2)
    gamma = a_1*(nu[0,0,:]*nu[1,0,:]+nu[1,1,:]*nu[0,1,:])
    epsilon = delta-norm
    x2 = (-gamma-np.sqrt(gamma**2-epsilon*beta))/beta
    a = np.array(([a_1,x2])).T
    return a

def nu_matrix(sigma_t_vector,identity_3d, corr_chole, N_equity, Nsim):
    S = sigma_t_vector.T*identity_3d
    n = S.T@corr_chole
    return n.transpose(1,2,0).reshape(N_equity,N_equity,Nsim)

def alpha_dot_nu(alpha_matrix,nu_matrix, N_equity, Nsim):
    a = np.sum((alpha_matrix.T*nu_matrix.transpose(1,0,2)),axis=1).T
    return a

def evolve(index_start, I_t, dS_S, Vola, alpha, target_vola, instant_interest_rate, dt, N_euler_grid, N_equity, Nsim, identity_3d, corr_chole):
    s = np.sum(alpha, axis=1)
    I_t_plus_1 = np.ones(dS_S[:,0,0].shape)*I_t
   # print("Start simulation ",index_start)
   # print("End simulation ", N_euler_grid+index_start+1)
    for t in range(index_start, N_euler_grid+index_start):
        if t==index_start:
            omega = target_vola
        else:
            nu = nu_matrix(Vola[:,t,:],identity_3d, corr_chole, N_equity, Nsim)
            omega = target_vola/np.linalg.norm(alpha_dot_nu(alpha, nu, N_equity, Nsim),axis=1)
        I_t_plus_1 = I_t_plus_1 * (1. + np.sum(alpha*dS_S[:,t,:],axis=1)*omega + dt * instant_interest_rate[t]*(1.-omega*s))
    return I_t_plus_1

def plot_regression(states, V_realized, V_alpha, I_t, alpha):
    plt.scatter(states,V_realized, label="Realized")
    plt.scatter(states,V_alpha, c='red', label='Regression')
    plt.title(r"$I_t$ = "+str(round(I_t,4))+" --- alpha = "+str(round(alpha,4)))
    plt.xlabel("State")
    plt.ylabel("$V_t$")
    plt.legend()
    plt.show()

    
Seed = 11
N_equity = 2
frequency = "month"
target_volatility = 5./100
I_0 = 1.
K = I_0     #option strike 
T = 2.      #option expiry
Nsim = 1e3  #number of MC paths
N_mc_forward = 1e6
number_I = 100
number_alpha=100
polynomial_basis = lambda x:  np.column_stack((x, x**2, x**3))


Nsim = int(Nsim)
N_mc_forward = int(N_mc_forward)
shape = (N_equity,N_equity,Nsim)
### Array of identity matrices for vectorized calculations
identity_3d = np.zeros(shape)
idx = np.arange(shape[0])
identity_3d[idx, idx, :] = 1

##Problem grids
I_grid = np.linspace(0.001*I_0,I_0*10,number_I)
alpha_grid = np.linspace(-2,2,number_alpha)
    
ACT = 365.0
if frequency == "month":
    month_dates = np.array([31.,28.,31.,30.,31.,30.,31.,31.,30.,31.,30.,31.])
    months = month_dates
    if T > 1.:
        for i in range(int(T)-1):
            months = np.append(months, month_dates)
    t_grid = np.cumsum(months)/ACT
    N_euler_grid = 60       
    state_index = np.arange(int(12*T)+1)*N_euler_grid
elif frequency == "day":
    t_grid = np.linspace(1./ACT, T, int(T*ACT))
    N_euler_grid = 2
    state_index = np.arange(int(365*T)+1)*N_euler_grid
t_grid = np.append(0.,t_grid)

##Reading Market Data and building model
names = ["DJ 50 EURO E","S&P 500 NET EUR"]
correlation = np.array(([1.,0.],[0.,1.]))
folder_name = "FakeSmilesDisplacedDiffusion"
D, F, V, LV = LoadFromTxt(names, folder_name)
spot_prices = np.array([])
for forward in F:
    spot_prices = np.append(spot_prices,forward.spot)
correlation_chole = np.linalg.cholesky(correlation)
model = LV_model(fixings=t_grid[1:], local_vol_curve=LV, forward_curve=F, N_grid = N_euler_grid)
euler_grid = model.time_grid
r_t = D.r_t(np.append(0.,euler_grid[:-1]))
dt_vector = model.dt
np_random = np.random

start_time = time.time()
##MC simulation of the risky assets 
np_random.seed(Seed)
S, simulations_Vola = model.simulate(corr_chole = correlation_chole, random_gen = np_random, normalization = 0, Nsim=Nsim)
S = np.insert(S,0,spot_prices,axis=1)
dS_S = (S[:,1:,:]-S[:,:-1,:])/S[:,:-1,:]

omega = target_volatility * np.ones(Nsim)   #since we parameterize the action such that the norm(alpha@nu)=1

Polynomial_grade = len(polynomial_basis(1.).T)
coeff_matrices = np.zeros((len(I_grid),len(t_grid), len(alpha_grid), Polynomial_grade+1))
V_t_plus = np.zeros((Nsim, len(I_grid)))
V_T = np.zeros((V_t_plus.shape))
state_index_rev = list(reversed(state_index[:-1]))
V_t = np.ones(V_t_plus.shape)*(-np.inf)

###Domande roberto: interpolazione a riga 137, un solo asset, che driver uso nella regressione?dS_S, S o W? che range di griglia uso per I e alpha
k = 0
i=0
for I_T in I_grid:
    V_T[:,i] = max(I_T-K,0.)
    i+=1
    
V_t_plus = V_T
for t in reversed(t_grid[:-1]):
    start_index = state_index_rev[k]
    index_I_grid = 0
    nu = nu_matrix(simulations_Vola[:,start_index,:],identity_3d, correlation_chole, N_equity, Nsim)
    for It in I_grid:
        alpha_index = 0
        for a in alpha_grid:
            alpha = alpha_vector(a,nu,Nsim)
            I_t_plus_1 = evolve(start_index, It, dS_S, simulations_Vola, alpha, target_volatility, r_t, dt_vector[-(k+1)], N_euler_grid, N_equity, Nsim, identity_3d, correlation_chole)
            V_realized = np.diag(interp1d(I_grid, V_t_plus, axis=1, fill_value="extrapolate")(I_t_plus_1))
            ##Regression
            states = np.log(S[:,start_index,0]/spot_prices[0]) #first asset 
            Y = polynomial_basis(states)
            Y = sm.add_constant(Y)
            ols = sm.OLS(V_realized, Y)
            ols_result = ols.fit()
            coeff_It_t = ols_result.params
            coeff_matrices[index_I_grid, k, alpha_index,:] = coeff_It_t   #saving regression coeff
            ##Continuation Value
            V_alpha = np.sum(coeff_It_t * Y, axis=1)      #continuation value
            V_t[:,index_I_grid] =  np.maximum(V_t[:,index_I_grid], V_alpha)
            #plot_regression(states, V_realized, V_alpha, It, a)
        
            alpha_index +=1
        index_I_grid +=1
    print(" Completed Month ",k," of ",T*12)
    print("Elapsed time ",(time.time()-start_time)/60., " min ")
    V_t_plus = V_t
    k += 1

V_0 = np.diag(interp1d(I_grid, V_t, axis=1)(np.ones(Nsim)*I_0))
np.save("Price_paths",V_t)
np.save("Regression_coeff",coeff_matrices)

print("--- Execution time: ", (time.time()-start_time)/60., " min ---")
I_t = np.ones(N_mc_forward)*I_0
np_random.seed(Seed+1)
S, simulations_Vola, dS_S = None, None, None
S, simulations_Vola = model.simulate(corr_chole = correlation_chole, random_gen = np_random, normalization = 0, Nsim=N_mc_forward)
S = np.insert(S,0,spot_prices,axis=1)
dS_S = (S[:,1:,:]-S[:,:-1,:])/S[:,:-1,:]

for i in range(len(t_grid[:-1])):
    V_max = np.ones(N_mc_forward)*(-np.inf)
    alpha_chosen = np.zeros(N_mc_forward)
    coeff_matrices_time = coeff_matrices[:,i,:,:]
    a_index = 0
    for a in alpha_grid:
        coeff_matrices_time_alpha = coeff_matrices_time[:,a_index,:]
        coefficients = interp1d(I_grid, coeff_matrices_time_alpha, axis=0, fill_value="extrapolate")(I_t) 
        a_index += 1
        states = np.log(S[:,state_index[i],0]/spot_prices[0]) #first asset 
        Y = polynomial_basis(states)
        Y = sm.add_constant(Y)
        expected_continuation = np.sum(coefficients * Y, axis=1)
        alpha_chosen[expected_continuation > V_max] = a
        V_max = expected_continuation
    nu = nu_matrix(simulations_Vola[:,state_index[i],:],identity_3d, correlation_chole, N_equity, N_mc_forward)
    alpha_best =  alpha_vector(alpha_chosen,nu,N_mc_forward)
    I_t = evolve(state_index[i], I_t, dS_S, simulations_Vola, alpha, target_volatility, r_t, dt_vector[i], N_euler_grid, N_equity, Nsim, identity_3d, correlation_chole)
