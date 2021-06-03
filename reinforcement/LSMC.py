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

#def alpha_vector(a1, nu, Nsim=1, norm=1):
 #   a_1 = np.ones(Nsim)*a1
  #  beta = nu[1,0,:]**2+nu[1,1,:]**2
   # delta = (a_1**2)*(nu[0,0,:]**2+nu[0,1,:]**2)
   # gamma = a_1*(nu[0,0,:]*nu[1,0,:]+nu[1,1,:]*nu[0,1,:])
   # epsilon = delta-norm
   # x2 = (-gamma-np.sqrt(gamma**2-epsilon*beta))/beta
   # print(x2)
   # a = np.array(([a_1,x2])).T
    #return a

def alpha_vector(a1, nu, Nsim=1, norm=1):
    a_1 = np.ones(Nsim)*a1
    beta = nu[1,0,:]**2 + nu[1,1,:]**2
    gamma = a_1*(nu[0,0,:]*nu[1,0,:]+ nu[0,1,:]*nu[1,1,:])
    delta = (a_1**2)*(nu[0,0,:]**2+nu[0,1,:]**2) - norm
    Delta = gamma**2-beta*delta
    Delta[Delta<0] = 0.
    a_2 = (-gamma+np.sqrt(Delta))/beta
    
    return np.array(([a_1,a_2])).T

def nu_matrix(sigma_t_vector,identity_3d, corr_chole, N_equity, Nsim):
    S = sigma_t_vector.T*identity_3d
    n = S.T@corr_chole
    return n.transpose(1,2,0).reshape(N_equity,N_equity,Nsim)

def alpha_dot_nu(alpha_matrix,nu_matrix, N_equity, Nsim):
    a = np.sum((alpha_matrix.T*nu_matrix.transpose(1,0,2)),axis=1).T
    return a

def evolve_TVS(index_start, I_t, dS_S, Vola, alpha, target_vola, instant_interest_rate, dt, N_euler_grid, N_equity, Nsim, identity_3d, corr_chole):
    s = np.sum(alpha, axis=1)
    I_t_plus_1 = np.ones(dS_S[:,0,0].shape)*I_t
    for t in range(index_start, N_euler_grid+index_start):
        if t==index_start:
            omega = target_vola
        else:
            nu = nu_matrix(Vola[:,t,:],identity_3d, corr_chole, N_equity, Nsim)
            omega = target_vola/np.linalg.norm(alpha_dot_nu(alpha, nu, N_equity, Nsim),axis=1)
        I_t_plus_1 = I_t_plus_1 * (1. + np.sum(alpha*dS_S[:,t,:],axis=1)*omega + dt * instant_interest_rate[t]*(1.-omega*s))
    return I_t_plus_1

def plot_regression(states, V_realized, V_alpha, I_t, alpha, time):
    plt.scatter(states,V_realized, label="Realized")
    plt.scatter(states,V_alpha, c='red', label='Regression')
    plt.title(r"$I_t$ = "+str(round(I_t,4))+" --- alpha = "+str(round(alpha,4))+" --- time = "+str(time))
    plt.xlabel("State")
    plt.ylabel("$V_t$")
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
        sigma = LV[i](t_in, logX_t[:,i])
        logX_t[:,i] = logX_t[:,i]-0.5*dt*(sigma**2)+sigma*np.sqrt(dt)*ep[i]
        S_t_plus_1[:,i] = np.exp(logX_t[:,i])*F[i](t_fin)
    return S_t_plus_1, sigma
    


Seed = 13
N_equity = 2
frequency = "month"
target_volatility = 5./100
I_0 = 1.
K = I_0     #option strike 
T = 2.      #option expiry
Nsim = 1e3  #number of MC paths
N_mc_forward = 1e6
print_logs=True
save_coeff = True
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
alpha_grid = np.linspace(-5.,5.,number_alpha)
    
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
names = ["S&P 500 NET EUR","DJ 50 EURO E"]
correlation = np.array(([1.,0.],[0.,1.]))
folder_name = "FakeSmiles"
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
            #print(alpha)
            I_t_plus_1 = evolve_TVS(start_index, It, dS_S, simulations_Vola, alpha, target_volatility, r_t, dt_vector[-(k+1)], N_euler_grid, N_equity, Nsim, identity_3d, correlation_chole)
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
        if(((It == I_grid[2]) or (It == I_grid[3]) or (It == I_grid[-1])) and (print_logs==True)):
            plot_regression(states, V_realized, V_alpha, It, a, t)
        
            alpha_index +=1
        index_I_grid +=1
    if print_logs:
        print(" Completed Month ",k," of ",T*12)
        print("Elapsed time ",(time.time()-start_time)/60., " min ")
    V_t_plus = V_t
    k += 1

if save_coeff:
    np.save("Price_paths",V_t)
    np.save("Regression_coeff",coeff_matrices)


if print_logs:
    t_LSMC = (time.time()-start_time)/60.
    print("--- Execution time for LSMC regression: ", t_LSMC, " min ---")
    print("Strating forward MC")
shape = (N_equity,N_equity,N_mc_forward)
### Array of identity matrices for vectorized calculations
identity_3d = np.zeros(shape)
idx = np.arange(shape[0])
identity_3d[idx, idx, :] = 1
I_t = np.ones(N_mc_forward)*I_0
S_t = np.ones((N_mc_forward,N_equity))*spot_prices
alpha_best = np.array([])
nu = np.array([])
for i in range(len(t_grid[:-1])):
    print("Month ",i)
    V_max = np.ones(N_mc_forward)*(-np.inf)
    alpha_chosen = np.zeros(N_mc_forward)
    coeff_matrices_time =  coeff_matrices[:,i,:,:]
    a_index = 0
    for a in alpha_grid:
        coeff_matrices_time_alpha = coeff_matrices_time[:,a_index,:]
        coefficients = interp1d(I_grid, coeff_matrices_time_alpha, axis=0, fill_value="extrapolate")(I_t) 
        a_index += 1
        states = np.log(S_t[:,0]/spot_prices[0]) #first asset 
        Y = polynomial_basis(states)
        Y = sm.add_constant(Y)
        expected_continuation = np.sum(coefficients * Y, axis=1)
        alpha_chosen[expected_continuation > V_max] = a
        V_max = expected_continuation
    dt = (t_grid[i+1]-t_grid[i])/N_euler_grid
    t_in = t_grid[i]
    for j in range(N_euler_grid):
        t_fin = t_in + dt
        dt = t_fin-t_in
        S_t_plus_1, sigma_t = evolve_martingale(S_t, LV, np_random, t_in, t_fin, correlation_chole, F, N_mc_forward, N_equity)
        nu = nu_matrix(sigma_t,identity_3d, correlation_chole, N_equity, N_mc_forward)
        if j==0:
            alpha_best =  alpha_vector(alpha_chosen,nu,N_mc_forward)
            s = np.sum(alpha_best,axis=1)
        dS_S = (S_t_plus_1-S_t)/S_t
        omega = target_volatility/np.linalg.norm(alpha_dot_nu(alpha_best, nu, N_equity, N_mc_forward),axis=1)
        I_t = I_t*( 1. + np.sum(alpha_best*dS_S,axis=1)*omega + dt * D.r_t(t_in)*(1.-omega*s))
        S_t = S_t_plus_1
        t_in = t_fin
        
discounted_payoff = Vanilla_PayOff(I_t, K)*D(T)

x, mean, err = MC_Analisys(discounted_payoff, 100)
plt.errorbar(x, mean,yerr=err, label="LSMC")
plt.axhline(y=0.04448152569910367, label="baseline",color="red")
plt.xlabel("MC throws")
plt.show()
print("Call option price ", np.mean(discounted_payoff))
print("error", np.std(discounted_payoff)/np.sqrt(N_mc_forward))
plt.hist(I_t, bins=20)
plt.xlabel("I_t")
plt.ylabel("Frequencies")

if print_logs:
    t_MC = (time.time()-t_LSMC*60.)/60.
    print("--- Execution time for forward MC: ", t_MC, " min ---")
    print("--- Total Execution time: ", t_MC + t_LSMC, " min ---")