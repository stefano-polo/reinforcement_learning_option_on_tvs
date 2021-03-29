from envs.pricing.pricing import DiscountingCurve, EquityForwardCurve, ForwardVariance, LocalVolatilityCurve
import numpy as np

def LoadFromTxt(asset_name, folder = None):
    dates_discounts, discounts = np.loadtxt(folder+"/discount_data.txt")
    D = DiscountingCurve(reference=0.,discounts=discounts,dates=dates_discounts)
    F, V, LV = [], [], []
    for name in asset_name:
        spot = np.loadtxt(folder+"/spot_data"+name+".txt")[0]
        repo_dates, repo_rates = np.loadtxt(folder+"/repo_data_"+str(name)+".txt")
        F.append(EquityForwardCurve(reference=0., discounting_curve = D, spot=spot, repo_rates=repo_rates,repo_dates=repo_dates,name=name,act="No"))
        
        spot_vola = np.loadtxt(folder+"/vola_data_"+str(name)+".txt")
        vola_strikes = np.loadtxt(folder+"/strikes_vola_data_"+str(name)+".txt")
        vola_dates = np.loadtxt(folder+"/maturities_vola_data_"+name+".txt")
        V.append(ForwardVariance(reference=0.,maturities=vola_dates,strikes=vola_strikes,spot_volatility=spot_vola,strike_interp=spot,act="No", name=name))
        
        lv_pars = np.loadtxt(folder+"/LV_param_data_"+str(name)+".txt")
        lv_money = np.loadtxt(folder+"/LV_money_vola_data_"+str(name)+".txt")
        lv_dates = np.loadtxt(folder+"/LV_maturities_vola_data_"+name+".txt")
        LV.append(LocalVolatilityCurve(lv_pars,lv_money,lv_dates,name))

    return D, F, V, LV