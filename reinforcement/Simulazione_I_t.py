import numpy as np 
import matplotlib.pyplot as plt

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
Nsim = 4e5 #number of MC paths
number_I = 50
number_alpha= 2


Nsim = int(Nsim)
shape = (N_equity,N_equity,Nsim)
### Array of identity matrices for vectorized calculations
identity_3d = np.zeros(shape)
idx = np.arange(shape[0])
identity_3d[idx, idx, :] = 1


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
names = ["DJ 50 EURO E","S&P 500 NET EUR"]
correlation = np.array(([1.,0.],[0.,1.]))
folder_name = "constant_vola_market"   #before_fake_smiles
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
    
##MC simulation of the risky assets 
np_random.seed(Seed)
S, simulations_Vola = model.simulate(corr_chole = correlation_chole, random_gen = np_random, normalization = 0, Nsim=Nsim)
S = np.insert(S,0,spot_prices,axis=1)
dS_S = (S[:,1:,:]-S[:,:-1,:])/S[:,:-1,:]

I_t = np.ones(Nsim)*I_0
alpha = np.ones((Nsim,N_equity))*np.array([1.,0.])
for i in range(0, len(euler_grid)):
    print(i)
    nu = nu_matrix(simulations_Vola[:,i,:],identity_3d, correlation_chole, N_equity, Nsim)
    omega = target_volatility/np.linalg.norm(alpha_dot_nu(alpha, nu, N_equity, Nsim),axis=1)
    I_t = I_t*(1. + omega*np.sum(alpha*dS_S[:,i,:],axis=1) + dt_vector[0]*r_t[i]*(1.-omega))
    
discounted_payoff = Vanilla_PayOff(I_t, K)*D(T)

x, mean, err = MC_Analisys(discounted_payoff, 100)
plt.errorbar(x, mean,yerr=err, label="MC")
plt.axhline(y=0.018160594508452997, label="Black optimal price",color="red")
plt.legend()
plt.xlabel("MC throws")
plt.show()