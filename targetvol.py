import numpy as np
from scipy import exp, sqrt, log, heaviside


def optimal_strategy(forward = None, variance = None, corr = None, Ntrials = 1):
    Ndim = len(forward)
    Ntrials = int(Ntrials)
    T = np.array([])     #temporal grid where the strategy change
    for i in range(Ndim):
        T = np.append(T,forward[i].T)
        T = np.append(T,variance[i].T)
    T = np.sort(np.asarray(list(set(T))))    #unique temporal structure
    
    print("Strategy time grid: ",T)
    mu = np.zeros((len(T),Ndim))      
    nu = np.zeros((Ndim,Ndim,len(T)))    #time dependent variance, covariance matrix  
    
    """Creating the mu time dependent vector and the nu time dependent matrix"""
    for i in range(len(T)):      
        vol = np.zeros(Ndim)
        for j in range(Ndim):
            vol[j] = sqrt(variance[j](T[i]))
            mu[i,j] = - (1./(T[i]-forward[j].reference)) * log((forward[j](T[i])*forward[j].discounting_curve(T[i]))/forward[j].spot)  #repo rates
        vol = np.identity(Ndim)*vol
        nu[:,:,i] = np.dot(np.dot(vol,corr),vol)
    
    alpha_t = np.zeros((len(T),Ndim))   #time dependent allocation strategy
    for i in range(len(T)):
        alpha = np.random.uniform(0,1,(Ntrials,Ndim))
        norm = np.sum(alpha,axis=1)
        norm = np.vstack((norm,norm))
        alpha = alpha/norm.T              #allocation strategy matrix (sum along each row is 1)
        f = np.dot(alpha,mu[i,:])/np.linalg.norm(np.dot(alpha,nu[:,:,i]),axis=1)  #score function vector
        print(f)
        alpha_t[i] = alpha[np.argmax(f)]
    
    return T,alpha_t