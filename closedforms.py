import numpy as np
import random as rnd
from scipy.interpolate import interp1d
import scipy.stats as si  #for gaussian cdf
from scipy import exp, log, sqrt


"""Functions for Black & Scholes Formula"""
def d1(S, K, T, r, q, sigma):
    """d_1 function for BS model"""
    return (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

def d2(S, K, T, r, q, sigma):
    """d_2 function for BS model"""
    return  (log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

def European_closed_form(S=None, K=None, T=None, r=None, q=None, sigma=None, typo = 1):
    """Closed form of Call Option for BS model"""
    d_1 = d1(S, K, T, r, q, sigma)
    d_2 = d2(S, K, T, r, q, sigma)
    if typo ==1:
        """Call option"""
        return S * exp(-q * T) * si.norm.cdf(d_1, 0.0, 1.0) - K * exp(-r * T) * si.norm.cdf(d_2, 0.0, 1.0)
    elif typo==-1:
        """Put option"""
        return K * exp(-r * T) * si.norm.cdf(-d_2, 0.0, 1.0) - S * exp(-q * T) * si.norm.cdf(-d_1, 0.0, 1.0)

def vega(S, K, T, r, q, sigma):
    d_1 = d1(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * np.sqrt(T) * si.norm.pdf(d_1)   
    
def implied_vol_newton(C,S,K,T,r,q,kind):
    xold = 0.
    xnew = 0.25#sqrt((2.*np.pi)/T)*(C/S)
    counter = 0
    tolerance = 0.0000001
    while abs(xnew-xold)>=tolerance:
        xold = xnew
        f = Black_closed_form(S, K, T, r, q, xold, kind)
        v = vega(S, K, T, r, q, xold)
        xnew = xold - (f-C)/v
        counter = counter+1
     
    return abs(xnew)


"""Geometric Average Asian Option"""
def sigma_z(m,sigma):
    return sigma * sqrt((2.*m+1)/(6.*(m+1)))

def rho_drift(m,r,sigma):
    vol_z = sigma_z(m,sigma)
    return 0.5 * (r - 0.5*(sigma**2)+vol_z**2)

def GMAsian_closed_form(m,S,K,T,r,q,sigma,typo):
    rho = rho_drift(m,r,sigma)
    vol_z = sigma_z(m,sigma)
    price = European_closed_form(S, K, T, rho, q, vol_z, typo)
    return price * exp((rho-r)*T)