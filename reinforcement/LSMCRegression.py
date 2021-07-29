import numpy as np 
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
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
     alpha[:,0] = 1. - angle
     alpha[:,1] = angle
     return alpha

Seed = 11
N_equity = 2
frequency = "month"
target_volatility = 5./100
I_0 = 1.
K = I_0     #option strike 
T = 1.      #option expiry
Nsim = 1e3 #number of Backguard paths
N_mc_forward = 7e5
print_logs=True
save_coeff = True
load_coeff = False
R = 1.
name = ""
number_I = 50
number_alpha= 2
Polynomial_grade = 1

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
    N_euler_grid = 1       
    state_index = np.arange(int(12*T)+1)*N_euler_grid
elif frequency == "day":
    t_grid = np.linspace(1./ACT, T, int(T*ACT))
    N_euler_grid = 2
    state_index = np.arange(int(365*T)+1)*N_euler_grid
t_grid = np.append(0.,t_grid)
n_times_minus_1 = len(t_grid[:-1])

##Reading Market Data and building model
names = ["DJ 50 EURO E","S&P 500 NET EUR"]
correlation = np.array(([1.,0.],[0.,1.]))
folder_name = "constant_vola_market"   #before_fake_smiles constant_vola_market
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
spot_prices = np.zeros(N_equity)

forwards = np.zeros((len(t_grid[1:]),N_equity))
for i, forward in enumerate(F):
    forwards[:,i] = forward(t_grid[:-1])
    spot_prices[i] = forward.spot
    
correlation_chole = np.linalg.cholesky(correlation)
model = LV_model(fixings=t_grid[1:], local_vol_curve=LV, forward_curve=F, N_grid = N_euler_grid)
euler_grid = model.time_grid
r_t = D.r_t(np.append(0.,euler_grid[:-1]))
dt_vector = model.dt
np_random = np.random

poly = PolynomialFeatures(degree=Polynomial_grade)
X = np.ones(N_equity)#np.ones((N_equity,N_equity))
X = X[:,np.newaxis]
X_ = poly.fit_transform(X)
coeff_matrices = np.zeros((number_I,n_times_minus_1,number_alpha, len(X_.T)))
    
start_time = time.time()
##MC simulation of the risky assets 
np_random.seed(Seed)
S, simulations_Vola = model.simulate(corr_chole = correlation_chole, random_gen = np_random, normalization = 0, Nsim=Nsim)
S = np.insert(S,0,spot_prices,axis=1)
dS_S = (S[:,1:,:]-S[:,:-1,:])/S[:,:-1,:]


V_t_plus = np.zeros((Nsim, number_I))
state_index_rev = list(reversed(state_index[:-1]))
V_t = np.ones(V_t_plus.shape)*(-np.inf)

for k, I_T in enumerate(I_grid):
    V_t_plus[:,k] = max(I_T-K,0.)


"""
t = list(reversed(t_grid[:-1]))[0]


V_realized_interpolator = interp1d(I_grid, V_t_plus, axis=1, fill_value="extrapolate")
start_index = state_index_rev[0]
#I_grid2 = np.array([I_grid[-5], I_grid[-1]])
for It in I_grid:
    print("I start ", It)
    alpha = compute_alphas(0, Nsim, N_equity)
    I_t_plus_1_optimal = evolve_TVS(start_index, It, dS_S, simulations_Vola, alpha, target_volatility, r_t, dt_vector[0], N_euler_grid, N_equity, Nsim, identity_3d, correlation_chole)
    V_realized_optimal = np.diag(V_realized_interpolator(I_t_plus_1_optimal))
    print("Mean I_t optimal ",np.mean(I_t_plus_1_optimal))
    print("Mean V_t optimal ",np.mean(V_realized_optimal))
    alpha = compute_alphas(1, Nsim, N_equity)
    I_t_plus_1_no_optimal = evolve_TVS(start_index, It, dS_S, simulations_Vola, alpha, target_volatility, r_t, dt_vector[0], N_euler_grid, N_equity, Nsim, identity_3d, correlation_chole)
    V_realized_no_optimal = np.diag(V_realized_interpolator(I_t_plus_1_no_optimal))
    print("Mean I_t NO optimal ",np.mean(I_t_plus_1_no_optimal))
    print("Mean V_t NO optimal ",np.mean(V_realized_no_optimal))
    if np.mean(V_realized_optimal)>np.mean(V_realized_no_optimal):
        print("GOOD \n")"""
        
for k, t in enumerate(reversed(t_grid[:-1])):
    start_index = state_index_rev[k]
    V_realized_interpolator = interp1d(I_grid, V_t_plus, axis=1, fill_value="extrapolate")
    state1 = S[:,start_index,0]#np.log(S[:,start_index,0]/forwards[ len(t_grid[:-1])-k-1,0])  #first asset 
    state2 =S[:,start_index,1]# np.log(S[:,start_index,1]/forwards[ len(t_grid[:-1])-k-1,1])
    Y = state1[:,np.newaxis]#np.column_stack((state1,state2))
    Y = poly.fit_transform(Y)
    for index_I_grid, It in enumerate(I_grid):
        V_t[:,index_I_grid] = -np.inf
        V_t_confronto = np.zeros((Nsim, number_alpha))
        V_alpha_confronto = np.zeros((Nsim,number_alpha))
        for angle_index, a in enumerate(angles_grid):
            alpha = compute_alphas(a, Nsim, N_equity)
            I_t_plus_1 = evolve_TVS(start_index, It, dS_S, simulations_Vola, alpha, target_volatility, r_t, dt_vector[0], N_euler_grid, N_equity, Nsim, identity_3d, correlation_chole)
            V_realized = np.diag(V_realized_interpolator(I_t_plus_1))
            mask = I_t_plus_1 >= K
            V_t_confronto[:,angle_index] = V_realized
            print("I_grid ",It)
            print("mean of I for angle_index ",angle_index," is ",np.mean(I_t_plus_1))
            print("mean of V for angle_index ",angle_index," is ",np.mean(V_realized))
            ##Regression
            reg = linear_model.LinearRegression(positive=True, normalize=False, fit_intercept=False, n_jobs=8).fit(Y,V_realized)
            coeff_It_t = reg.coef_
            coeff_matrices[index_I_grid, len(t_grid[:-1])-k-1, angle_index, :] = coeff_It_t
            ###Continuation value
            V_alpha = reg.predict(Y)#np.sum(coeff_It_t * Y, axis=1)     ##continuation value
            V_alpha_confronto[:,angle_index] = V_alpha#[0]
            V_t[:,index_I_grid] = np.maximum(V_t[:,index_I_grid],V_realized)
            print("Coeff ", coeff_It_t)
            if (print_logs) and (angle_index == number_alpha-1):
                plt.scatter(state1, V_t_confronto[:,0], color="orange",label="V_realized optimal")
                plt.scatter(state1, V_alpha_confronto[:,0], color="red", label="Regression optimal")
                plt.scatter(state1, V_t_confronto[:,1], color="blue",label="V_realized")
                plt.scatter(state1, V_alpha_confronto[:,1], color="black", label="Regression")
                plt.xlabel("State 1")
                plt.ylabel("$V_t$")
                plt.title(r"$I_t$ = "+str(round(It,4))+" --- time = "+str(round(t,3)))
                plt.legend()
                plt.show()
    
    V_t_plus = V_t
    if print_logs:
        print(" Completed Month ",k," of ",T*12)
        print("Elapsed time ",(time.time()-start_time)/60., " min ")

            
## Appunti 29/007
#L'inserimento dei dati grezzi e eseguendo una semplice Z score normalization di scikit learn sembra aver migliorato il secondo 
# step di iterazione. Da verificare come trasportare questo elemento anche nella simulazione forward. Il problema tuttavia si riscontra
# nella terza iterazione. La cosa funziona per un asset funziona se e solo se si massimizza il V realized e non il V_alpha
# da provare questo con 2 asset inseriti e in più mettere il cut se si raggiungono valori negativi 