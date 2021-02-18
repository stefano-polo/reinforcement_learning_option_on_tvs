import numpy as np
from numpy import log, exp, sqrt
import xml.etree.ElementTree as ET
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, Black, ForwardVariance, ACT_365, LocalVolatilityCurve

def load_fake_market_lv():
    correlation = np.array(([1,0.5],[0.5,1]))       #correlation matrix
    index_equity = [5]

    tree = ET.parse('../TV_example.xml')
    root = tree.getroot()
    N_stocks = len(root[3][0][1][1][1][0])
    names = np.array([])
    for i in range(N_stocks):
        names=np.append(names,root[3][0][1][1][1][0][i].text)
    delete_equity = [0,11,12,13,14,15]
    names = np.delete(names,delete_equity)
    reference_date = float(root[3][1][0][0][2][0].text)

    spot_prices = np.zeros(len(index_equity))
    j = 0
    for i in index_equity:
        if i>=6:
            spot_prices[j] = float(root[3][i+3][0][1][0][0].text)
        else:
            spot_prices[j] = float(root[3][3+i][0][0][1][0].text)
        j = j+1
        
    discounts = np.zeros(len(root[3][1][0][0][0][0][0]))
    discounts_dates = np.zeros(len(root[3][1][0][0][0][0][0]))
    for i in range(len(root[3][1][0][0][0][0][0])):
        discounts_dates[i] = float(root[3][1][0][0][0][0][0][i].text)
        discounts[i] = float(root[3][1][0][0][0][1][0][i].text)
    D = DiscountingCurve(reference=reference_date, discounts=discounts, dates=discounts_dates, act="365")

    F = []
    max_dates = np.array([10])
    index = 0
    for i in index_equity:
        if i>=6:
            repo_dates = np.array([reference_date+1, max(max_dates)])
            repo_rates = np.zeros(2)
            F.append(EquityForwardCurve(reference=reference_date, discounting_curve=D, repo_dates=repo_dates,repo_rates=repo_rates, spot=spot_prices[index],act="360"))
        else:
            repo_dates = np.zeros(len(root[3][3+i][0][0][0][0][0]))
            repo_rates = np.zeros(len(root[3][3+i][0][0][0][1][0]))
            for j in range (len(root[3][3+i][0][0][0][0][0])):
                repo_dates[j] = float(root[3][3+i][0][0][0][0][0][j].text)
                repo_rates[j] = float(root[3][3+i][0][0][0][1][0][j].text)
            max_dates = np.append(max_dates,max(repo_dates))
            F.append(EquityForwardCurve(reference=reference_date, discounting_curve=D, repo_dates=repo_dates,repo_rates=repo_rates, spot=spot_prices[index],act="360"))
        index = index+1
    F = F[0]

    for i in index_equity:
        vola_dates = np.zeros(len(root[3][i+3][1][2][0][0]))
        vola_strikes = np.zeros(len(root[3][i+3][1][2][2][0]))
        for j in range(len(root[3][i+3][1][2][0][0])):
            vola_dates[j] = float(root[3][i+3][1][2][0][0][j].text)
        for j in range(len(root[3][i+3][1][2][2][0])):
            vola_strikes[j] = float(root[3][i+3][1][2][2][0][j].text)
        market_vola = np.zeros(len(root[3][i+3][1][2][0][0])*len(root[3][i+3][1][2][2][0]))
        for k in range (len(market_vola)):
            market_vola[k] = float(root[3][i+3][1][2][1][0][k].text)
        market_vola = np.reshape(market_vola,(len(vola_dates),len(vola_strikes))).T
    vola_maturities = ACT_365(vola_dates, reference_date)
    sigma_LV = LocalVolatilityCurve(market_vola,vola_strikes,vola_maturities)
    sigma_LV.parameterization_with_h(F)
    sigma_LV2 = LocalVolatilityCurve(market_vola,vola_strikes,vola_maturities)
    sigma_LV2.parameterization_with_h(F)
    F_all = [F,F]
    sigmaLV = [sigma_LV,sigma_LV2]
    return D, F_all, sigmaLV, correlation