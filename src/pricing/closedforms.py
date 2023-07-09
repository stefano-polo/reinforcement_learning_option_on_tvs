import sys
from typing import List, Union

import numpy as np
import scipy.stats as si  # for gaussian cdf
from numpy import exp, log, sqrt

sys.path.insert(1, "./src")

from pricing.pricing import EquityForwardCurve


def d1(
    forward: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    time_to_maturity: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Black and Scholes d1 closed-form
    :param forward (Union[float, np.ndarray]): forward price
    :param strike (Union[float, np.ndarray]) : strike price
    :param time_to_maturity (Union[float, np.ndarray]) : time to maturity expressed in yrf
    :param volatility (Union[float, np.ndarray]) : volatility
    :return: d1 (Union[float, np.ndarray])
    """
    return (log(forward / strike) + 0.5 * (volatility**2) * time_to_maturity) / (
        volatility * sqrt(time_to_maturity)
    )


def BS_European_option_closed_form(
    forward: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    time_to_maturity: Union[float, np.ndarray],
    discount: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray],
    call_put: Union[float, np.ndarray] = 1.0,
) -> Union[float, np.ndarray]:
    """
    Black and Scholes closed-form for European options
    :param forward (Union[float, np.ndarray]): forward price
    :param strike (Union[float, np.ndarray]) : strike price
    :param time_to_maturity (Union[float, np.ndarray]) : time to maturity expressed in yrf
    :param discount (Union[float, np.ndarray]) : discount factor
    :param volatility (Union[float, np.ndarray]) : volatility
    :param call_put (float) : 1 for call, -1 for put
    :return: Black and Scholes option price (Union[float, np.ndarray])
    """
    if type(forward) == np.ndarray:
        assert len(forward) == len(time_to_maturity) == len(discount)
    std = sqrt(time_to_maturity) * volatility
    d_1 = call_put * d1(forward, strike, time_to_maturity, volatility)
    d_2 = d_1 - call_put * std
    return (
        discount
        * call_put
        * (forward * si.norm.cdf(d_1, 0.0, 1.0) - strike * si.norm.cdf(d_2, 0.0, 1.0))
    )


def Price_to_BS_ImpliedVolatility(
    time_to_maturity: float,
    forward: float,
    strike: float,
    option_price: float,
    call_put: float,
    discount: float,
) -> float:
    """
    Function that calculates the implied volatility of a European option using the Black and Scholes closed-form. The function exploits the implementation of
    the algorthm developed by Peter Jäckel from the library lets_be_rational (open source) or the pyGem package of Intesa Sanpaolo FE (closed source).
    :param time_to_maturity (float): time to maturity expressed in yrf
    :param forward (float): forward price
    :param strike (float): strike price
    :param option_price (float): option price
    :param call_put (float or int): 1 for call, -1 for put
    :param discount (float): discount factor
    :return (float): implied volatility
    """
    assert type(option_price) in [float, np.float64, np.float32]
    assert type(forward) in [float, np.float64, np.float32]
    assert type(strike) in [float, np.float64, np.float32]
    assert type(time_to_maturity) in [float, np.float64, np.float32]
    assert type(discount) in [float, np.float64, np.float32]
    assert type(call_put) in [int, float, np.float64, np.float32, np.int32, np.int64]
    try:
        from pyGem.GemUtility import BS_ImpliedVol

        return BS_ImpliedVol(
            time_to_maturity, forward, strike, option_price, call_put, discount
        )
    except ImportError:
        try:
            import lets_be_rational.LetsBeRational as lbr

            return lbr.implied_volatility_from_a_transformed_rational_guess(
                option_price / discount, forward, strike, time_to_maturity, call_put
            )
        except ImportError:
            try:
                from py_vollib.black_scholes.implied_volatility import implied_volatility
                option_flag = "c" if call_put >= 1.0 else "p"
                zero_interest_rate = -1.0 * np.log(discount) / time_to_maturity
                spot = forward * discount
                return implied_volatility(price=option_price, S=spot, K=strike, t=time_to_maturity, r=zero_interest_rate, flag=option_flag)
            except:
                raise ImportError(
                    "pyGem and LetsBeRational are not installed. Please install pyGem and LetsBeRational to use this function."
                )


def volatility_asian(
    n_averages: Union[int, np.ndarray], volatility: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Function that calculates the volatility of a Geometric average Asian option.
    :param n_averages (int): number of averaging dates of the option
    :param volatility (Union[float, np.ndarray]): volatility of the underlying asset
    :return: volatility of the Geometric average Asian option (Union[float, np.ndarray])
    """
    return volatility * sqrt((2.0 * n_averages + 1) / (6.0 * (n_averages + 1)))


def interest_rate_asian(
    n_averages: Union[int, np.ndarray],
    zero_interest_rate: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Function that calculates the interest rate of a Geometric average Asian option.
    :param n_averages (Union[int, np.ndarray]): number of averaging dates of the option
    :param zero_interest_rate (Union[float, np.ndarray]): zero interest rate
    :param volatility (Union[float, np.ndarray]): volatility of the underlying asset
    :return: interest rate of the Geometric average Asian option (Union[float, np.ndarray])
    """
    vol_asian = volatility_asian(n_averages, volatility)
    return 0.5 * (zero_interest_rate - 0.5 * (volatility**2) + vol_asian**2)


def GA_Asian_option_closed_form(
    forward: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    time_to_maturity: Union[float, np.ndarray],
    discount: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray],
    n_averages: Union[int, np.ndarray] = 10,
    call_put: Union[float, np.ndarray] = 1.0,
) -> Union[float, np.ndarray]:
    """
    Function that calculates the price of a Geometric average Asian option using the Black and Scholes closed-form.
    :param forward (Union[float, np.ndarray]): forward price
    :param strike (Union[float, np.ndarray]): strike price
    :param time_to_maturity (Union[float, np.ndarray]): time to maturity expressed in yrf
    :param discount (Union[float, np.ndarray]): discount factor
    :param volatility (Union[float, np.ndarray]): volatility of the underlying asset
    :param n_averages (Union[int, np.ndarray]): number of averaging dates of the option
    :param call_put (Union[float, np.ndarray]): 1 for call, -1 for put
    :return: price of the Geometric Average Asian option (Union[float, np.ndarray])
    """
    zero_interest_rate = (-1.0 / time_to_maturity) * log(discount)
    r_asian = interest_rate_asian(n_averages, zero_interest_rate, volatility)
    forward_asian = forward * exp((r_asian - zero_interest_rate) * time_to_maturity)
    vol_asian = volatility_asian(n_averages, volatility)
    return BS_European_option_closed_form(
        forward_asian, strike, time_to_maturity, discount, vol_asian, call_put
    )


def volatility_basket(
    volatility_assets: np.ndarray, correlation_matrix: np.ndarray
) -> np.ndarray:
    """
    Function that calculates the volatility of a Geometric Average basket option.
    :param volatility_assets (np.ndarray): volatility of the underlying assets
    :param correlation_matrix (np.ndarray): correlation matrix of the underlying assets
    :return: volatility of the Geometric Average basket option (float)
    """
    assert (
        len(volatility_assets) == len(correlation_matrix) == len(correlation_matrix.T)
    )
    n_assets = len(volatility_assets)
    vol_matrix = np.identity(n_assets) * volatility_assets
    covariance_matrix = (vol_matrix @ correlation_matrix) @ vol_matrix
    return (1.0 / n_assets) * sqrt(np.sum(covariance_matrix))


def forward_basket(
    forwards_list: List[EquityForwardCurve],
    volatility_assets: np.ndarray,
    correlation_matrix: np.ndarray,
    time_to_maturity: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Function that calculates the forward price of a Geometric Average basket option at time_to_maturity.
    :param forwards_list (List[EquityForwardCurve]): list of forward curves of the underlying assets
    :param volatility_assets (np.ndarray): volatility of the underlying assets
    :param correlation_matrix (np.ndarray): correlation matrix of the underlying assets
    :param time_to_maturity (Union[float, np.ndarray]): time to maturity expressed in yrf
    :return: forward price of the Geometric Average basket option (Union[float, np.ndarray]) at time_to_maturity
    """
    assert len(forwards_list) == len(volatility_assets)
    fwd_basket = 1.0
    sigma_basket = volatility_basket(volatility_assets, correlation_matrix)
    n_assets = len(forwards_list)
    for i in range(n_assets):
        fwd_basket = (
            fwd_basket
            * forwards_list[i](time_to_maturity)
            * exp((-0.5 * volatility_assets[i] ** 2) * time_to_maturity)
        )
    fwd_basket = fwd_basket ** (1.0 / n_assets)
    fwd_basket = fwd_basket * exp(0.5 * (sigma_basket**2) * time_to_maturity)
    return fwd_basket


def GAM_Basket_option_closed_form(
    forwards_list: List[EquityForwardCurve],
    strike: Union[float, np.ndarray],
    time_to_maturity: Union[float, np.ndarray],
    discount: Union[float, np.ndarray],
    volatility_array: np.array,
    correlation: np.ndarray,
    call_put: Union[float, np.ndarray] = 1.0,
) -> Union[float, np.ndarray]:
    """
    Function that calculates the price of a Geometric Average basket option using the Black and Scholes closed-form.
    :param forwards_list (list[EquityForwardCurve]): list of forward curves of the underlying assets
    :param strike (Union[float, np.ndarray]): strike price
    :param time_to_maturity (Union[float, np.ndarray]): time to maturity expressed in yrf
    :param discount (Union[float, np.ndarray]): discount factor
    :param volatility_array (np.ndarray): volatility of the underlying assets
    :param correlation (np.ndarray): correlation matrix of the underlying assets
    :param call_put (Union[float, np.ndarray]): 1 for call, -1 for put
    :return: price of the Geometric Average basket option (Union[float, np.ndarray])
    """
    vol_basket = volatility_basket(volatility_array, correlation)
    basket_fwd = forward_basket(
        forwards_list, volatility_array, correlation, time_to_maturity
    )
    return BS_European_option_closed_form(
        basket_fwd, strike, time_to_maturity, discount, vol_basket, call_put
    )
