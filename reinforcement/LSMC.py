import numpy as np 
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, LV_model, ForwardVariance, quad_piecewise
from envs.pricing.targetvol import Drift
from envs.pricing.loadfromtxt import LoadFromTxt
from envs.pricing.targetvol import Markowitz_solution
from envs.pricing.closedforms import European_option_closed_form

def alpha_vector(a1, nu, norm=1):
    beta = nu[1,0]**2+nu[1,1]**2
    delta = (a1**2)*(nu[0,0]**2+nu[0,1]**2)
    gamma = a1*(nu[0,0]*nu[1,0]+nu[1,1]*nu[0,1])
    epsilon = delta-norm
    x2 = (-gamma-np.sqrt(gamma**2-epsilon*beta))/beta
    return np.array([a1,x2])

def nu_matrix(sigma_t_vector,identity_3d, corr_chole, N_equity, Nsim):
    S = sigma_t_vector.T*identity_3d
    n = S.T@corr_chole
    return n.transpose(1,2,0).reshape(N_equity,N_equity,Nsim)

def alpha_dot_nu(alpha_matrix,nu_matrix, N_equity, Nsim):
    a = np.sum(alpha_matrix*nu_matrix.transpose(1,2,0),axis=2)
    return a.transpose(1,0)
    
N_equity = 2
frequency = "month"
target_volatility = 5./100
I_0 = 1.
K = I_0     #option strike 
T = 2.      #option expiry
Nsim = 5  #number of MC paths


I_grid = np.linspace(0.,I_0*10,100)
alpha_grid = np.linspace(-5,5,100)

shape = (N_equity,N_equity,Nsim)
identity_3d = np.zeros(shape)
idx = np.arange(shape[0])
identity_3d[idx, idx, :] = 1

identity_3d

Identity = np.identity(N_equity)
Nsim = int(Nsim)
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

names = ["DJ 50 EURO E","S&P 500 NET EUR"]
correlation = np.array(([1.,0.],[0.,1.]))
folder_name = "FakeSmilesDisplacedDiffusion"
D, F, V, LV = LoadFromTxt(names, folder_name)
correlation_chole = np.linalg.cholesky(correlation)



V_T = np.zeros((Nsim, len(I_grid)))
V_t = np.zeros((Nsim, len(I_grid)))
i=0
for I_t in I_grid:
    V_T[:,i] = max(I_t-K,0.)
    i+=1

for t in reversed(t_grid[:-1]):
    for It in I_grid:
        V_t = -np.inf
        for a in alpha_grid:
            alpha = alpha_vector
        


idx = 4

sigma = np.array(([40,55],[10,25],[80,95],[5,45],[50,55]))/100
corr = np.array(([1.,0.7],[0.7,1.]))
corr_chole = np.linalg.cholesky(corr)
nu = nu_matrix(sigma,identity_3d, corr_chole, N_equity,Nsim)
prod = alpha_dot_nu(sigma,nu, N_equity, Nsim)
print("MIO ",nu[:,:,idx])
print("Mio ",np.sqrt(np.sum(nu[:,:,idx]*nu[:,:,idx],axis=1)))
print("Mio prod ",prod[idx])
print("\n")
s = sigma[idx]
S = s*np.identity(N_equity)
f = S@corr_chole
print("Corretto", np.sqrt(np.sum(f*f,axis=1)))
print("Corretto ",f)
print("corretto prod", s@f)
