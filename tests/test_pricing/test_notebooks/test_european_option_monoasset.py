import sys

import numpy as np

sys.path.insert(1, "./src")
from params import *

from pricing.closedforms import (
    BS_European_option_closed_form,
    Price_to_BS_ImpliedVolatility,
)
from pricing.montecarlo import MC_Data_Blocking
from pricing.pricing import (
    Black,
    DiscountingCurve,
    EquityForwardCurve,
    ForwardVariance,
    Vanilla_PayOff,
)


def test_convergence(
    discounting_curve: DiscountingCurve,
    forward_curve: EquityForwardCurve,
    variance_cuve: ForwardVariance,
):
    D = discounting_curve
    F = forward_curve
    V = variance_cuve
    kind = 1
    maturities = np.arange(1, 31) * 0.15
    B_model = Black(maturities, forward_curve=F, variance_curve=V, sampling=sampling)
    S_t = B_model.simulate(n_sim=N_simulations, seed=12)
    X_t = S_t / F(maturities)
    E_X = np.mean(X_t, axis=0)
    K_norm = 1
    pay_normalized = Vanilla_PayOff(X_t, K_norm, kind, sampling)  # ATM forward pricing

    """Calculating closed form"""
    call_black = BS_European_option_closed_form(
        F(maturities), F(maturities), maturities, D(maturities), volatility, kind
    )
    dates_index = [1, 5, 10, 29]
    for n in dates_index:
        _, result, result_err = MC_Data_Blocking(pay_normalized[:, n], N_block)
        mean_price = result * D(maturities[n]) * F(maturities[n])
        err_price = result_err * D(maturities[n]) * F(maturities[n])
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
        if vola > volatility:
            assert vola - 2.55 * y_lower[-1] <= volatility
        else:
            assert vola + 2.55 * y_upper[-1] >= volatility
