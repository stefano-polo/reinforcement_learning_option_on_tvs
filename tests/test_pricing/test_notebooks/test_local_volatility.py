import sys

sys.path.insert(1, "./src")

import numpy as np

from pricing.closedforms import Price_to_BS_ImpliedVolatility
from pricing.montecarlo import MC_results
from pricing.pricing import LocalVolatilityCurve, LV_model, Vanilla_PayOff
from pricing.read_market import LoadFromTxt


def test_interpolation_mkt_data_monoasset():
    idx = 0
    n_sigma = 2.55
    folder = "./src/market_data/Basket10Assets/"
    asset_names_list = (
        "DJ 50 TR",
        "S&P 500 NET EUR",
        "MSCI EM MKT EUR",
        "I NKY NTR EUR",
        "FTSE100 NTR E",
        "SMI TR EUR",
        "DAX 30 E",
        "FTSEMIBN",
        "HSI NTR EUR",
    )
    _, F, V, LV, _ = LoadFromTxt(
        asset_names_list, folder, "ATM_FWD", local_vol_model=True
    )
    F = F[idx]
    V = V[idx]
    LV = LV[idx]
    assert V.asset_name == F.asset_name
    assert LV.asset_name == F.asset_name

    spot_volatilities = V.market_volatility.T
    vola_strikes = V.K
    maturities = np.loadtxt(folder + "maturities_vola_data_" + F.asset_name + ".txt")
    money_ness = np.zeros((len(vola_strikes), len(maturities)))
    for j in range(len(vola_strikes)):
        for i in range(len(maturities)):
            money_ness[j, i] = vola_strikes[j] / F(maturities[i])

    IV_curve = LocalVolatilityCurve(
        spot_volatilities, money_ness, maturities, V.asset_name, "piecewise"
    )

    N_grid = 160
    sampling = "antithetic"
    N_simulation = 1e5
    expiries = LV.T
    model = LV_model(
        fixings=expiries,
        local_vol_curve=LV,
        forward_curve=F,
        n_euler_grid=N_grid,
        sampling=sampling,
        return_grid_values_for_tvs=False,
    )
    X_t = model.simulate(n_sim=N_simulation, seed=21) / model.forward_values

    kind = 1
    expiries = LV.T
    np.exp(LV.log_moneyness)
    moneyness_matrix = np.exp(LV.log_moneyness)
    n_strikes = len(moneyness_matrix)
    n_expiries = len(expiries)
    imp_volatility = np.zeros((n_strikes, n_expiries))
    imp_volatility_plus = np.zeros((n_strikes, n_expiries))
    imp_volatility_minus = np.zeros((n_strikes, n_expiries))
    n_expiries = len(expiries)
    E_X = np.mean(X_t, axis=0)
    for j in range(n_expiries):
        for i in range(n_strikes):
            k = moneyness_matrix[i, j]
            if k >= 1:
                kind = 1  # 1 buono
            elif k < 1:
                kind = -1  # -1 buono
            option = Vanilla_PayOff(X_t, k, kind, sampling)
            result, err_result = MC_results(option)
            imp_volatility[i, j] = Price_to_BS_ImpliedVolatility(
                expiries[j], E_X[j], k, result[j], kind, 1.0
            )
            imp_volatility_plus[i, j] = Price_to_BS_ImpliedVolatility(
                expiries[j], E_X[j], k, result[j] + err_result[j], kind, 1.0
            )
            imp_volatility_minus[i, j] = Price_to_BS_ImpliedVolatility(
                expiries[j], E_X[j], k, result[j] - err_result[j], kind, 1.0
            )

    y_lower = np.zeros(((n_strikes, n_expiries)))
    y_upper = np.zeros(((n_strikes, n_expiries)))
    for i in range(n_strikes):
        for j in range(n_expiries):
            if imp_volatility_minus[i][j] < imp_volatility_plus[i][j]:
                y_lower[i][j] = abs(imp_volatility[i][j] - imp_volatility_minus[i][j])
                y_upper[i][j] = abs(imp_volatility_plus[i][j] - imp_volatility[i][j])
            elif imp_volatility_minus[i][j] > imp_volatility_plus[i][j]:
                y_lower[i][j] = abs(imp_volatility[i][j] - imp_volatility_plus[i][j])
                y_upper[i][j] = abs(imp_volatility_minus[i][j] - imp_volatility[i][j])

    for i in range(n_expiries):
        err_lower = n_sigma * y_lower.T[i]
        err_upper = n_sigma * y_upper.T[i]
        this_moneyness = moneyness_matrix[:, i]
        expected_volas = IV_curve(expiries[i], np.log(this_moneyness))
        simulated = imp_volatility[:, i]
        # Check compatibility results
        for j in range(len(expected_volas)):
            expected = expected_volas[j]
            computed = simulated[j]
            error_up = err_upper[j]
            error_low = err_lower[j]
            if computed > expected:
                assert computed - 2.55 * error_low <= expected
            else:
                assert computed + 2.55 * error_up >= expected
