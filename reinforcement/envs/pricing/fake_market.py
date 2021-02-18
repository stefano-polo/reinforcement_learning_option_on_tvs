import numpy as np
from numpy import log, exp, sqrt
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, Black, ForwardVariance

def load_fake_market(N_equity, r, maturity):
    reference =0
    if N_equity ==3:
        correlation = np.array(([1,0.5,0.3],[0.5,1,0.3],[0.3,0.3,1]))       #correlation matrix
        spot_prices = np.array([100,200,89])
        T_repo3 = np.array([2/12.,5/12.,maturity])
        repo_rate3 = np.array([0.02,0.02,0.02])/10
    else:
        correlation = np.array(([1,0.5],[0.5,1]))       #correlation matrix
        spot_prices = np.array([100,200])

    r_t = np.array([r,r,r,r])
    T_discounts = np.array([0.,3/12,4/12.,maturity])      #data observation of the market discounts factor
    market_discounts = exp(-r_t*T_discounts)       #market discounts factor
    repo_rate1 = np.array([0.22,0.22,0.22])/10
    T_repo1 = np.array([1/12,4./12,maturity])       #data observation of the market repo rates for equity 1
    repo_rate2 = np.array([0.72,0.42,0.02])/10  #market repo rates for equity 1 0.52
    T_repo2 = np.array([1/12.,4/12.,maturity])

    sigma1 = np.array([20,20.,20.])/100
    T_sigma1 = np.array([2/12,5./12,maturity])
    K1 = np.array([spot_prices[0],500])
    spot_vola1 = np.array((sigma1,sigma1))                                      #market implied volatility matrix
    sigma2 = np.array([20,20,20])/100
    T_sigma2 =  np.array([2/12.,6/12,maturity])
    K2 = np.array([spot_prices[1],600])
    spot_vola2 = np.array((sigma2,sigma2))
    if N_equity==3:
        sigma3 = np.array([10,10,10])/100
        T_sigma3 =  np.array([2/12.,6/12,maturity])
        K3 = np.array([spot_prices[2],600])
        spot_vola3 = np.array((sigma3,sigma3))

    D = DiscountingCurve(reference=reference, discounts=market_discounts,dates=T_discounts)
    F = []
    V = []
    q = repo_rate1
    T_q = T_repo1
    s_vola = spot_vola1
    T_vola = T_sigma1
    K = K1
    F.append(EquityForwardCurve(reference=reference,spot=spot_prices[0],discounting_curve=D,repo_dates=T_q,repo_rates=q))
    V.append(ForwardVariance(reference=reference,spot_volatility=s_vola,strikes=K,maturities=T_vola,strike_interp=spot_prices[0]))
    q = repo_rate2
    T_q = T_repo2
    s_vola = spot_vola2
    T_vola = T_sigma2
    K = K2
    F.append(EquityForwardCurve(reference=reference,spot=spot_prices[1],discounting_curve=D,repo_dates=T_q,repo_rates=q))
    V.append(ForwardVariance(reference=reference,spot_volatility=s_vola,strikes=K,maturities=T_vola,strike_interp=spot_prices[1]))
    if N_equity==3:

        q = repo_rate3
        T_q = T_repo3
        s_vola = spot_vola3
        T_vola = T_sigma3
        K = K3
        F.append(EquityForwardCurve(reference=reference,spot=spot_prices[2],discounting_curve=D,repo_dates=T_q,repo_rates=q))
        V.append(ForwardVariance(reference=reference,spot_volatility=s_vola,strikes=K,maturities=T_vola,strike_interp=spot_prices[2]))

    return D, F, V, correlation, spot_prices
