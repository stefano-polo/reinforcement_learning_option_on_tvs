import sys

import numpy as np
import pytest

sys.path.insert(1, "./src")
from pricing.closedforms import (
    BS_European_option_closed_form,
    GA_Asian_option_closed_form,
    GAM_Basket_option_closed_form,
    Price_to_BS_ImpliedVolatility,
)


@pytest.mark.parametrize(
    "call_put, expected_result",
    [
        (1.0, 54.42),
        (-1.0, 1.93),
    ],
)
def test_bs_model_plain_vanilla_analytical_price(
    call_put: float, expected_result: float
):
    """
    Benchmark taken from https://goodcalculators.com/black-scholes-calculator/
    """
    spot = 300.0
    strike = 250.0
    time_to_maturity = 1.0
    zero_interest_rate = 0.01
    volatility = 0.15
    discount_factor = np.exp(-zero_interest_rate * time_to_maturity)
    forward_value = spot / discount_factor
    assert BS_European_option_closed_form(
        forward_value, strike, time_to_maturity, discount_factor, volatility, call_put
    ) == pytest.approx(expected_result, 0.1)


def test_bs_model_call_put_parity() -> None:
    spot = 300.0
    strike = 250.0
    time_to_maturity = 1.0
    zero_interest_rate = 0.01
    volatility = 0.15
    discount_factor = np.exp(-zero_interest_rate * time_to_maturity)
    forward_value = spot / discount_factor
    call_price = BS_European_option_closed_form(
        forward_value, strike, time_to_maturity, discount_factor, volatility, 1.0
    )
    put_price = BS_European_option_closed_form(
        forward_value, strike, time_to_maturity, discount_factor, volatility, -1.0
    )
    put_price_from_parity = call_price + (strike - forward_value) * discount_factor
    assert put_price == pytest.approx(put_price_from_parity, 0.0001)


@pytest.mark.parametrize(
    "call_put, iv",
    [
        (1.0, 0.15),
        (-1.0, 0.15),
    ],
)
def test_from_price_to_iv(call_put: float, iv: float):
    """
    Benchmark taken from https://goodcalculators.com/black-scholes-calculator/
    """

    spot = 300.0
    strike = 250.0
    time_to_maturity = 1.0
    zero_interest_rate = 0.01
    discount_factor = np.exp(-zero_interest_rate * time_to_maturity)
    forward_value = spot / discount_factor
    price = BS_European_option_closed_form(
        forward_value, strike, time_to_maturity, discount_factor, iv, call_put
    )
    assert Price_to_BS_ImpliedVolatility(
        time_to_maturity, forward_value, strike, price, call_put, discount_factor
    ) == pytest.approx(iv, 0.001)


@pytest.mark.parametrize(
    "call_put, expected_result",
    [
        (1.0, 50.49),
        (-1.0, 0.115),
    ],
)
def test_geometric_average_asian_option_pricing(
    call_put: float, expected_result: float
):
    spot = 300.0
    strike = 250.0
    time_to_maturity = 1.0
    zero_interest_rate = 0.01
    volatility = 0.15
    discount_factor = np.exp(-zero_interest_rate * time_to_maturity)
    forward_value = spot / discount_factor
    number_of_avg_points = 10
    assert GA_Asian_option_closed_form(
        forward_value,
        strike,
        time_to_maturity,
        discount_factor,
        volatility,
        number_of_avg_points,
        call_put,
    ) == pytest.approx(expected_result, 0.01)
