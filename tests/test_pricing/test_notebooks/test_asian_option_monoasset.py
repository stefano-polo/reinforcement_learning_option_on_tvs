import sys

import numpy as np
import pytest
from scipy.stats.mstats import gmean

sys.path.insert(1, "./src")
from params import *

from pricing.closedforms import (
    GA_Asian_option_closed_form,
    Price_to_BS_ImpliedVolatility,
    interest_rate_asian,
    volatility_asian,
)
from pricing.montecarlo import MC_Data_Blocking
from pricing.pricing import (
    Black,
    DiscountingCurve,
    EquityForwardCurve,
    ForwardVariance,
    Vanilla_PayOff,
)

N_averages = 30
maturity = 4.0
dates = np.linspace(0.0, maturity, N_averages)
volatility = 0.5
asian_vola = volatility_asian(N_averages, volatility)
sampling = "antithetic"


@pytest.fixture
def variance_cuve_ga() -> ForwardVariance:
    K_spot_vola = np.array([spot_price, 200])
    spot_vol = np.array(([volatility, volatility], [volatility, volatility]))
    spot_vol_dates = np.array([0.1, T_max])
    return ForwardVariance(
        reference=t,
        maturity_dates=spot_vol_dates,
        strikes=K_spot_vola,
        market_volatility_matrix=spot_vol,
        strike_interp=spot_price,
    )


def test_convergence(
    discounting_curve: DiscountingCurve,
    forward_curve: EquityForwardCurve,
    variance_cuve_ga: ForwardVariance,
):
    D = discounting_curve
    F = forward_curve
    V = variance_cuve_ga
    B_model = Black(dates, forward_curve=F, variance_curve=V, sampling=sampling)
    kind = 1
    S_t = B_model.simulate(n_sim=N_simulations, seed=18)
    G_mean = gmean(S_t, axis=1)
    forward_g_asian = spot_price * np.exp(
        interest_rate_asian(N_averages, r, volatility) * maturity
    )
    X_t = G_mean / forward_g_asian
    E_X = np.mean(X_t)
    K_norm = 1
    pay_normalized = Vanilla_PayOff(
        X_t, K_norm, call_put=kind, sampling=sampling
    )  # ATM forward pricing

    """Calculating closed form"""
    expected_value = GA_Asian_option_closed_form(
        F(maturity),
        forward_g_asian,
        maturity,
        D(maturity),
        volatility,
        N_averages,
        kind,
    )
    _, result, result_err = MC_Data_Blocking(pay_normalized, N_block)
    mean_price = result * D(maturity) * forward_g_asian
    err_price = result_err * D(maturity) * forward_g_asian
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
            maturity, E_X, 1.0, result[i], kind, 1.0
        )
        imp_volatility_plus[i] = Price_to_BS_ImpliedVolatility(
            maturity, E_X, 1.0, result[i] + result_err[i], kind, 1.0
        )
        imp_volatility_minus[i] = Price_to_BS_ImpliedVolatility(
            maturity, E_X, 1.0, result[i] - result_err[i], kind, 1.0
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
    if vola > asian_vola:
        assert vola - 2.55 * y_lower[-1] <= asian_vola
    else:
        assert vola + 2.55 * y_upper[-1] >= asian_vola
