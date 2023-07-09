import sys

import numpy as np
import pytest

sys.path.insert(1, "./src")
from params import *

from pricing.pricing import DiscountingCurve, EquityForwardCurve, ForwardVariance


@pytest.fixture
def discounting_curve() -> DiscountingCurve:
    zero_interest_rate = np.array([r, r, r])
    zero_interest_rate_dates = np.array([0.0, 5, T_max])
    d = np.exp(
        -zero_interest_rate * zero_interest_rate_dates
    )  # market discount factors
    return DiscountingCurve(
        reference=t, discounts=d, discount_dates=zero_interest_rate_dates
    )


@pytest.fixture
def forward_curve(discounting_curve: DiscountingCurve) -> EquityForwardCurve:
    return EquityForwardCurve(
        t,
        spot_price,
        discounting_curve,
        repo_dates=np.array([0.0, T_max]),
        repo_rates=np.array([0.1 / 100, 0.1 / 100]),
    )


@pytest.fixture
def variance_cuve() -> ForwardVariance:
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
