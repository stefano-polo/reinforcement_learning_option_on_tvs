import numpy as np 
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, LV_model, ForwardVariance, quad_piecewise
from envs.pricing.targetvol import Drift
from envs.pricing.loadfromtxt import LoadFromTxt
from envs.pricing.targetvol import Markowitz_solution
from envs.pricing.closedforms import European_option_closed_form
from scipy.interpolate import interp1d

def alpha_vector(a1, nu, Nsim=1, norm=1):
    a_1 = np.ones(Nsim)*a1
    beta = nu[1,0]**2+nu[1,1]**2
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

def evolve(index_start, I_t, dS_S, alpha, omega, instant_interest_rate, dt, N_euler_grid):
    s = np.sum(alpha, axis=1)
    I_t_plus_1 = np.ones(dS_S[:,0,0].shape)*I_t
    for t in range(index_start, N_euler_grid):
        I_t_plus_1 = I_t_plus_1 * (1. + np.sum(alpha*dS_S[:,t,:],axis=1)*omega + dt[index_start] * instant_interest_rate[t]*(1.-omega*s))
    return I_t_plus_1
    
Seed = 11
N_equity = 2
frequency = "month"
target_volatility = 5./100
I_0 = 1.
K = I_0     #option strike 
T = 2.      #option expiry
Nsim = 3  #number of MC paths
number_I = 2
number_alpha=2


I_grid = np.linspace(0.,I_0*10,number_I)
alpha_grid = np.linspace(-2,2,number_alpha)

Nsim = int(Nsim)
shape = (N_equity,N_equity,Nsim)
identity_3d = np.zeros(shape)
idx = np.arange(shape[0])
identity_3d[idx, idx, :] = 1

identity_3d

Identity = np.identity(N_equity)


t_grid = np.array([0.,1.,2.])
N_euler_grid = 3
state_index = np.arange(len(t_grid))*N_euler_grid
print("State index", state_index)
#ACT = 365.0
#if frequency == "month":
    #month_dates = np.array([31.,28.,31.,30.,31.,30.,31.,31.,30.,31.,30.,31.])
    #months = month_dates
    #if T > 1.:
      #  for i in range(int(T)-1):
     #       months = np.append(months, month_dates)
    #t_grid = np.cumsum(months)/ACT
    #N_euler_grid = 60       
    #state_index = np.arange(int(12*T)+1)*N_euler_grid
#elif frequency == "day":
    ##t_grid = np.linspace(1./ACT, T, int(T*ACT))
   ## N_euler_grid = 2
 ##   state_index = np.arange(int(365*T)+1)*N_euler_grid
#t_grid = np.append(0.,t_grid)

names = ["DJ 50 EURO E","S&P 500 NET EUR"]
correlation = np.array(([1.,0.],[0.,1.]))
folder_name = "FakeSmilesDisplacedDiffusion"
D, F, V, LV = LoadFromTxt(names, folder_name)
spot_prices = np.array([])
for forward in F:
    spot_prices = np.append(spot_prices,forward.spot)
    
print("Spot prices", spot_prices)
correlation_chole = np.linalg.cholesky(correlation)
model = LV_model(fixings=t_grid[1:], local_vol_curve=LV, forward_curve=F, N_grid = N_euler_grid)
euler_grid = model.time_grid
r_t = D.r_t(np.append(0.,euler_grid[:-1]))
dt_vector = model.dt
np_random = np.random

##MC simulation of the risky assets 
np_random.seed(Seed)
S, simulations_Vola = model.simulate(corr_chole = correlation_chole, random_gen = np_random, normalization = 0, Nsim=Nsim)
S = np.insert(S,0,spot_prices,axis=1)
dS_S = (S[:,1:,:]-S[:,:-1,:])/S[:,:-1,:]

V_t_plus = np.zeros((Nsim, len(I_grid)))
i=0
for I_t in I_grid:
    V_t_plus[:,i] = max(I_t-K,0.)
    i+=1


omega = target_volatility * np.ones(Nsim)   #since we parameterize the action such that the norm(alpha@nu)=1

sigma = np.array(([20,10],[30,50],[90,30],[11,22],[99,11]))/100.
nu = nu_matrix(simulations_Vola[:,0,:],identity_3d, correlation_chole, N_equity, Nsim)

k = 0
state_index_rev = list(reversed(state_index[:-1]))
for t in reversed(t_grid[:-1]):
    start_index = state_index_rev[k]
    k += 1
    for It in I_grid:
        V_t = -np.inf
        for a in alpha_grid:
            alpha = alpha_vector(a,nu,Nsim)
            I_t_plus_1 = evolve(start_index, It, dS_S, alpha, omega, r_t, dt_vector, N_euler_grid)
            V_realized = interp1d(I_grid, V_t_plus, fill_value="extrapolate")(I_t_plus_1)
            print(V_realized)