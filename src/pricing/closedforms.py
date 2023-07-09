import numpy as np
import scipy.stats as si  # for gaussian cdf
from numpy import exp, log, sqrt

""" Closed-form pricing for Black-Scholes-Merton model """


def d1(
    forward: float or np.ndarray,
    strike: float or np.ndarray,
    time_to_maturity: float or np.ndarray,
    volatility: float or np.ndarray,
) -> float or np.ndarray:
    """
    Black and Scholes d1 closed-form
    :param forward (float or np.ndarray[float]): forward price
    :param strike (float or np.ndarray[float]) : strike price
    :param time_to_maturity (float or np.ndarray[float]) : time to maturity expressed in yrf
    :param volatility (float or np.ndarray[float]) : volatility
    :return: d1 (float or np.ndarray[float])
    """
    return (log(forward / strike) + 0.5 * (volatility**2) * time_to_maturity) / (
        volatility * sqrt(time_to_maturity)
    )


def BS_European_option_closed_form(
    forward: float or np.ndarray,
    strike: float or np.ndarray,
    time_to_maturity: float or np.ndarray,
    discount: float or np.ndarray,
    volatility: float or np.ndarray,
    call_put: float = 1.0,
) -> float or np.ndarray:
    """
    Black and Scholes closed-form for European options
    :param forward (float or np.ndarray[float]): forward price
    :param strike (float or np.ndarray[float]) : strike price
    :param time_to_maturity (float or np.ndarray[float]) : time to maturity expressed in yrf
    :param discount (float or np.ndarray[float]) : discount factor
    :param volatility (float or np.ndarray[float]) : volatility
    :param call_put (float) : 1 for call, -1 for put
    :return: Black and Scholes option price (float or np.ndarray[float])
    """
    if type(forward) == np.ndarray:
        assert len(forward) == len(time_to_maturity) == len(discount)
    std = sqrt(time_to_maturity) * volatility
    d_1 = call_put * d1(forward, strike, time_to_maturity, volatility)
    d_2 = call_put * (d_1 - std)
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
        except:
            raise ImportError(
                "pyGem and LetsBeRational are not installed. Please install pyGem and LetsBeRational to use this function."
            )


def volatility_asian(n_averages: int, volatility: float) -> float:
    """
    Function that calculates the volatility of a Geometric average Asian option.
    :param n_averages (int): number of averaging dates of the option
    :param volatility (float): volatility of the underlying asset
    :return: volatility of the Geometric average Asian option (float)
    """
    return volatility * sqrt((2.0 * n_averages + 1) / (6.0 * (n_averages + 1)))


def interest_rate_asian(
    n_averages: int, zero_interest_rate: float, volatility: float
) -> float:
    """
    Function that calculates the interest rate of a Geometric average Asian option.
    :param n_averages (int): number of averaging dates of the option
    :param zero_interest_rate (float): zero interest rate
    :param volatility (float): volatility of the underlying asset
    :return: interest rate of the Geometric average Asian option (float)
    """
    vol_asian = volatility_asian(n_averages, volatility)
    return 0.5 * (zero_interest_rate - 0.5 * (volatility**2) + vol_asian**2)


def GA_Asian_option_closed_form(
    forward: float,
    strike: float,
    time_to_maturity: float,
    discount: float,
    volatility: float,
    n_averages: int = 10,
    call_put: float = 1.0,
) -> float:
    """
    Function that calculates the price of a Geometric average Asian option using the Black and Scholes closed-form.
    :param forward (float): forward price
    :param strike (float): strike price
    :param time_to_maturity (float): time to maturity expressed in yrf
    :param discount (float): discount factor
    :param volatility (float): volatility of the underlying asset
    :param n_averages (int): number of averaging dates of the option
    :param call_put (float): 1 for call, -1 for put
    :return: price of the Geometric Average Asian option (float)
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
) -> float:
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
    forwards_list: list,
    volatility_assets: np.ndarray,
    correlation_matrix: np.ndarray,
    time_to_maturity: float,
) -> float:
    """
    Function that calculates the forward price of a Geometric Average basket option at time_to_maturity.
    :param forwards_list (list[EquityForwardCurve]): list of forward curves of the underlying assets
    :param volatility_assets (np.ndarray): volatility of the underlying assets
    :param correlation_matrix (np.ndarray): correlation matrix of the underlying assets
    :param time_to_maturity (float): time to maturity expressed in yrf
    :return: forward price of the Geometric Average basket option (float) at time_to_maturity
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
    forwards_list: list,
    strike: float,
    time_to_maturity: float,
    discount: float,
    volatility_array: np.array,
    correlation: np.ndarray,
    call_put: float = 1.0,
) -> float:
    """
    Function that calculates the price of a Geometric Average basket option using the Black and Scholes closed-form.
    :param forwards_list (list[EquityForwardCurve]): list of forward curves of the underlying assets
    :param strike (float): strike price
    :param time_to_maturity (float): time to maturity expressed in yrf
    :param discount (float): discount factor
    :param volatility_array (np.ndarray): volatility of the underlying assets
    :param correlation (np.ndarray): correlation matrix of the underlying assets
    :param call_put (float): 1 for call, -1 for put
    :return: price of the Geometric Average basket option (float)
    """
    vol_basket = volatility_basket(volatility_array, correlation)
    basket_fwd = forward_basket(
        forwards_list, volatility_array, correlation, time_to_maturity
    )
    return BS_European_option_closed_form(
        basket_fwd, strike, time_to_maturity, discount, vol_basket, call_put
    )
