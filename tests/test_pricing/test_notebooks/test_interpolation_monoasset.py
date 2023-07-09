import sys

sys.path.insert(1, "./src")

import numpy as np

from pricing.closedforms import Price_to_BS_ImpliedVolatility
from pricing.montecarlo import MC_results
from pricing.pricing import Black, Vanilla_PayOff
from pricing.read_market import LoadFromTxt


def test_interpolation_mkt_data_monoasset():
    idx = 0
    folder = "./src/market_data/Basket10Assets/"
    asset_names_list = (
        "DJ 50 TR",
        "S&P 500 NET EUR",
        "MSCI EM MKT EUR",
        "I NKY NTR EUR",
        "DAX 30 E",
        "FTSE100 NTR E",
        "SMI TR EUR",
        "FTSEMIBN",
        "CAC 40 NTR",
        "HSI NTR EUR",
    )
    _, F, V, _ = LoadFromTxt(asset_names_list, folder, "ATM_FWD", local_vol_model=False)
    F = F[idx]
    V = V[idx]
    assert V.asset_name == F.asset_name
    N_simulations = 6e5
    sampling = "antithetic"
    call_put = 1
    maturities = np.loadtxt(folder + "maturities_vola_data_" + F.asset_name + ".txt")
    BS = Black(maturities, forward_curve=F, variance_curve=V, sampling=sampling)
    S_t = BS.simulate(n_sim=N_simulations, seed=14)
    X_t = S_t / F(maturities)
    strike = 1.0
    pay = Vanilla_PayOff(X_t, strike, call_put, sampling=sampling)
    result, err = MC_results(pay)
    E = np.mean(X_t, axis=0)
    T = maturities
    imp_volatility = np.zeros(len(T))
    imp_volatility_plus = np.zeros(len(T))
    imp_volatility_minus = np.zeros(len(T))
    for i in range(len(T)):
        imp_volatility[i] = Price_to_BS_ImpliedVolatility(
            T[i], E[i], strike, result[i], call_put, 1.0
        )
        imp_volatility_minus[i] = Price_to_BS_ImpliedVolatility(
            T[i], E[i], strike, result[i] - err[i], call_put, 1.0
        )
        imp_volatility_plus[i] = Price_to_BS_ImpliedVolatility(
            T[i], E[i], strike, result[i] + err[i], call_put, 1.0
        )

    y_lower = np.zeros(len(T))
    y_upper = np.zeros(len(T))
    for i in range(len(T)):
        if imp_volatility_minus[i] < imp_volatility_plus[i]:
            y_lower[i] = abs(imp_volatility[i] - imp_volatility_minus[i])
            y_upper[i] = abs(imp_volatility_plus[i] - imp_volatility[i])
        elif imp_volatility_minus[i] > imp_volatility_plus[i]:
            y_lower[i][j] = abs(imp_volatility[i][j] - imp_volatility_plus[i])
            y_upper[i][j] = abs(imp_volatility_minus[i] - imp_volatility[i])

    # Check compatibility results
    for i in range(len(V.spot_vol)):
        expected = V.spot_vol[i]
        computed = imp_volatility[i]
        error_up = y_upper[i]
        error_low = y_lower[i]
        if computed > expected:
            assert computed - 2.55 * error_low <= expected
        else:
            assert computed + 2.55 * error_up >= expected
