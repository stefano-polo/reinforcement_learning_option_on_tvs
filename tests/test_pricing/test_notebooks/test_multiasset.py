import sys

import numpy as np

sys.path.insert(1, "./src")
from scipy.stats.mstats import gmean

from pricing.closedforms import (
    GAM_Basket_option_closed_form,
    Price_to_BS_ImpliedVolatility,
    forward_basket,
    volatility_basket,
)
from pricing.montecarlo import MC_Data_Blocking
from pricing.pricing import (
    Black,
    DiscountingCurve,
    EquityForwardCurve,
    ForwardVariance,
    Vanilla_PayOff,
)

t = 0
spot_price = np.array([110.0, 120.0, 97.0, 133.0])
T_max = 10
N_simulation = 1e4
N_block = 100
volatility = np.array([20.0, 30.0, 25.0, 32.0]) / 100
corr = np.array(
    (
        [1, 0.15, 0.10, 0.20],
        [0.15, 1.0, -0.05, 0.18],
        [0.1, -0.05, 1.0, 0.13],
        [0.20, 0.18, 0.13, 1],
    )
)
sampling = "antithetic"
N_equity = len(corr)
sigma_basket = volatility_basket(volatility, corr)


def test_convergence(discounting_curve: DiscountingCurve):
    D = discounting_curve
    F = []  # list of all equity forward curves
    repos = np.array([0.5, 0.9, 0.38, 0.44])
    for i in range(N_equity):
        F.append(
            EquityForwardCurve(
                reference=t,
                spot=spot_price[i],
                discounting_curve=D,
                repo_rates=np.array([repos[i], repos[i]]) / 100,
                repo_dates=np.array([0.0, T_max]),
            )
        )
    T1 = np.array([0.08, 0.17, 0.25, 0.33, 0.42, 0.50, 1.0, 2.0, T_max])
    K1 = np.array([spot_price[0], 164, 172, 180])

    sigma1 = np.array(
        (
            [
                volatility[0],
                volatility[0],
                volatility[0],
                volatility[0],
                volatility[0],
                volatility[0],
                volatility[0],
                volatility[0],
                volatility[0],
            ],
            [32, 29.1, 28.9, 29.3, 29.4, 29.4, 29.6, 30.6, 30.63],
            [33.6, 29.3, 29, 29.3, 29.3, 29.3, 29.3, 30.6, 30.59],
            [35, 29.7, 29.4, 29.5, 29.4, 29.3, 29.3, 30.5, 30.46],
        )
    )
    sigma1 = sigma1

    T2 = np.array([0.08, 0.17, 0.25, 0.33, 0.42, 0.50, 1.0, 2.0, T_max])
    K2 = np.array([spot_price[1], 164, 172, 180])
    sigma2 = np.array(
        (
            [
                volatility[1],
                volatility[1],
                volatility[1],
                volatility[1],
                volatility[1],
                volatility[1],
                volatility[1],
                volatility[1],
                volatility[1],
            ],
            [32, 29.1, 28.9, 29.3, 29.4, 29.4, 29.6, 30.6, 30.63],
            [33.6, 29.3, 29, 29.3, 29.3, 29.3, 29.3, 30.6, 30.59],
            [35, 29.7, 29.4, 29.5, 29.4, 29.3, 29.3, 30.5, 30.46],
        )
    )
    sigma2 = sigma2

    T3 = np.array([0.08, 0.17, 0.25, 0.33, 0.42, 0.50, 1.0, 2.0, T_max])
    K3 = np.array([spot_price[2], 164, 172, 180])
    sigma3 = np.array(
        (
            [
                volatility[2],
                volatility[2],
                volatility[2],
                volatility[2],
                volatility[2],
                volatility[2],
                volatility[2],
                volatility[2],
                volatility[2],
            ],
            [32, 29.1, 28.9, 29.3, 29.4, 29.4, 29.6, 30.6, 30.63],
            [33.6, 29.3, 29, 29.3, 29.3, 29.3, 29.3, 30.6, 30.59],
            [35, 29.7, 29.4, 29.5, 29.4, 29.3, 29.3, 30.5, 30.46],
        )
    )

    T4 = np.array([0.08, 0.17, 0.25, 0.33, 0.42, 0.50, 1.0, 2.0, T_max])
    K4 = np.array([spot_price[3], 164, 172, 180])
    sigma4 = np.array(
        (
            [
                volatility[3],
                volatility[3],
                volatility[3],
                volatility[3],
                volatility[3],
                volatility[3],
                volatility[3],
                volatility[3],
                volatility[3],
            ],
            [32, 29.1, 28.9, 29.3, 29.4, 29.4, 29.6, 30.6, 30.63],
            [33.6, 29.3, 29, 29.3, 29.3, 29.3, 29.3, 30.6, 30.59],
            [35, 29.7, 29.4, 29.5, 29.4, 29.3, 29.3, 30.5, 30.46],
        )
    )

    V = []  # list of variances
    V.append(
        ForwardVariance(
            reference=t,
            market_volatility_matrix=sigma1.T,
            strikes=K1,
            maturity_dates=T1,
            strike_interp=spot_price[0],
        )
    )
    V.append(
        ForwardVariance(
            reference=t,
            market_volatility_matrix=sigma2.T,
            strikes=K2,
            maturity_dates=T2,
            strike_interp=spot_price[1],
        )
    )
    V.append(
        ForwardVariance(
            reference=t,
            market_volatility_matrix=sigma3.T,
            strikes=K3,
            maturity_dates=T3,
            strike_interp=spot_price[2],
        )
    )
    V.append(
        ForwardVariance(
            reference=t,
            market_volatility_matrix=sigma4.T,
            strikes=K4,
            maturity_dates=T4,
            strike_interp=spot_price[3],
        )
    )

    maturities = np.arange(1, 31) * 0.15
    model = Black(
        maturities,
        forward_curve=F,
        variance_curve=V,
        correlation_matrix=corr,
        sampling=sampling,
    )
    S_t = model.simulate(n_sim=N_simulation, seed=10)
    G_mean = gmean(S_t, axis=2)
    maturities = np.arange(1, 31) * 0.15
    kind = 1
    f = forward_basket(F, volatility, corr, maturities)
    X_t = G_mean / f
    E_X = np.mean(X_t, axis=0)
    K_norm = 1
    pay_normalized = Vanilla_PayOff(
        X_t, K_norm, kind, sampling=sampling
    )  # ATM forward pricing

    """Calculating closed form"""
    call_black = GAM_Basket_option_closed_form(
        F, f, maturities, D(maturities), volatility, corr, kind
    )

    for n in [12, 20, 26]:
        _, result, result_err = MC_Data_Blocking(pay_normalized[:, n], N_block)
        mean_price = result * D(maturities[n]) * f[n]
        err_price = result_err * D(maturities[n]) * f[n]
        expected_value = call_black[n]
        price = mean_price[-1]
        error = err_price[-1] * 2.55
        if price > expected_value:
            assert price - error <= expected_value
        else:
            assert price + error >= expected_value

        imp_volatility_mean = np.zeros(N_block)
        imp_volatility_plus = np.zeros(N_block)
        imp_volatility_minus = np.zeros(N_block)
        for i in range(N_block):
            imp_volatility_mean[i] = Price_to_BS_ImpliedVolatility(
                maturities[n], E_X[n], 1.0, result[i], kind, 1.0
            )
            imp_volatility_plus[i] = Price_to_BS_ImpliedVolatility(
                maturities[n], E_X[n], 1.0, result[i] + result_err[i], kind, 1.0
            )
            imp_volatility_minus[i] = Price_to_BS_ImpliedVolatility(
                maturities[n], E_X[n], 1.0, result[i] - result_err[i], kind, 1.0
            )

        y_lower = np.zeros(N_block)
        y_upper = np.zeros(N_block)
        for i in range(N_block):
            if imp_volatility_minus[i] < imp_volatility_plus[i]:
                y_lower[i] = abs(imp_volatility_mean[i] - imp_volatility_minus[i])
                y_upper[i] = abs(imp_volatility_plus[i] - imp_volatility_mean[i])
            elif imp_volatility_minus[i] > imp_volatility_plus[i]:
                y_lower[i] = abs(imp_volatility_mean[i] - imp_volatility_plus[i])
                y_upper[i] = abs(imp_volatility_minus[i] - imp_volatility_mean[i])

        vola = imp_volatility_mean[-1]
        if vola > sigma_basket:
            assert vola - 2.55 * y_lower[-1] <= sigma_basket
        else:
            assert vola + 2.55 * y_upper[-1] >= sigma_basket
