import numpy as np
import scipy.stats as si  #for gaussian cdf
from numpy import exp, log, sqrt


"""Functions for Black & Scholes Formula"""
def d1(forward = None, strike= None, maturity=None, reference=None, volatility=None):
    """d_1 function for BS model"""
    return (log(forward / strike) + 0.5 * (volatility ** 2) * (maturity-reference)) / (volatility * sqrt(maturity-reference))

def d2(forward = None, strike= None, maturity=None, reference=None, volatility=None):
    """d_1 function for BS model"""
    return (log(forward / strike) - 0.5 * (volatility ** 2) * (maturity-reference)) / (volatility * sqrt(maturity-reference))

def European_option_closed_form(forward = None, strike= None, maturity=None, reference=None, zero_interest_rate = None, volatility=None, typo = 1):
    """Closed form of Call Option for BS model"""
    d_1 = d1(forward, strike, maturity, reference, volatility)
    d_2 = d2(forward, strike, maturity, reference, volatility)
    discount = exp(-zero_interest_rate*(maturity-reference))
    if typo ==1:
        """Call option"""
        return discount * (forward * si.norm.cdf(d_1, 0.0, 1.0) - strike * si.norm.cdf(d_2, 0.0, 1.0))
    elif typo==-1:
        """Put option"""
        return discount * (strike * si.norm.cdf(-d_2, 0.0, 1.0) - forward * si.norm.cdf(-d_1, 0.0, 1.0))

def Delta(forward = None, strike= None, maturity=None, reference=0, volatility=None):
    return si.norm.cdf(d1(forward, strike, maturity, reference, volatility))

def StrikeFromDelta(forward=None, maturity=None, reference=None, delta=None, volatility=None):
    return forward(maturity)*np.exp(0.5*(volatility**2)*(maturity-reference)-volatility*np.sqrt(maturity-reference)*si.norm.ppf(delta))

def Vega(forward=None, strike=None, maturity=None, reference=None, dividends=None, volatility=None):
    d_1 = d1(forward, strike, maturity, reference, volatility)
    return S * np.exp(-dividends * (maturity-reference)) * np.sqrt(maturity-reference) * si.norm.pdf(d_1)

def implied_vol_newton(market_price=None, forward=None, strike=None, maturity=None, reference=None, zero_interest_rate=None, dividends=None, volatility=None, kind=None):
    """Newton algorithm for implied volatility"""
    xold = 0.
    xnew = 0.25
    counter = 0
    tolerance = 0.0000001
    while abs(xnew-xold)>=tolerance:
        xold = xnew
        f = European_option_closed_form(forward,strike,maturity,reference,zero_interest_rate,xold,typo)
        v = Vega(forward,strike,maturity,reference,dividends,xold)
        xnew = xold - (f-market_price)/v
        counter = counter+1

    return abs(xnew)


"""Geometric Average Asian Option"""
def volatility_asian(N_averages=None, volatility = None):
    return volatility * sqrt((2.*N_averages+1)/(6.*(N_averages+1)))

def interest_rate_asian(N_averages=None, zero_interest_rate=None, volatility=None):
    vol_asian = volatility_asian(N_averages,volatility)
    return 0.5 * (zero_interest_rate - 0.5*(volatility**2)+vol_asian**2)

def GA_Asian_option_closed_form(forward = None, strike= None, maturity=None, reference=None, zero_interest_rate = None, volatility=None, N_averages=None, typo = 1):
    r_asian = interest_rate_asian(N_averages,zero_interest_rate,volatility)
    forward_asian = forward*exp((r_asian-zero_interest_rate)*(maturity-reference))
    vol_asian = volatility_asian(N_averages,volatility)
    return European_option_closed_form(forward_asian, strike, maturity, reference, zero_interest_rate, vol_asian, typo)



"""Geometric averaged basket option in d dimensions"""
def volatility_basket(volatility=None, correlation=None):
    S = np.identity(len(volatility))*volatility
    S = np.dot(np.dot(S,correlation),S)
    return (1/len(volatility))*sqrt(np.sum(S))

def forward_basket(forward=None, volatility=None, correlation=None, maturity=None, reference=None):
    F_basket = 1
    sigma_basket = volatility_basket(volatility,correlation)
    for i in range (len(forward)):
        F_basket = F_basket*forward[i](maturity)*exp((-0.5*volatility[i]**2)*maturity)
    F_basket = F_basket**(1/len(forward))
    F_basket = F_basket*exp(0.5*(sigma_basket**2)*maturity)
    return F_basket

def GAM_Basket_option_closed_form(forward = None, strike= None, maturity=None, reference=None, zero_interest_rate = None, volatility=None, correlation=None, typo = 1):
    vol_basket = volatility_basket(volatility,correlation)
    F =  forward_basket(forward,volatility,correlation,maturity,reference)
    return European_option_closed_form(F, strike, maturity, reference, zero_interest_rate, vol_basket, typo)
