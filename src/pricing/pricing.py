from numpy import exp, log, sqrt
from pricing_utility import *
from scipy.interpolate import PchipInterpolator, interp1d

"""Classes for my simulation"""


class Curve:
    def __init__(self, **kwargs):
        raise Exception("do not instantiate this class.")

    def __call__(self, date):  # return the value of the curve at a defined time
        return self.curve(date)


class DiscountingCurve(Curve):
    """Discounting curve"""

    def __init__(
        self,
        reference: int or float,
        discounts: np.ndarray,
        discount_dates: np.ndarray,
        day_count_convention: str = None,
    ) -> None:
        """
        Market Discounting Curve.
        The discounting curve is the curve that is used to discount future cash flows.
        :param reference (int or float): reference date (in days or yrf)
        :param discounts (np.ndarray[float]): market discount factors
        :param discount_dates (np.ndarray[int] or np.ndarray[float]): dates of the market discount factors (in days or yrf)
        :param day_count_convention (str, default=None): day count convention (available ACT_365, ACT_360 or None). If None then the discount_dates and reference are assumed to be in yrf.
        """
        self.reference = reference
        assert (
            discounts.ndim == discount_dates.ndim == 1
        )  # discount and discount_dates must be 1D arrays
        assert len(discounts) == len(
            discount_dates
        )  # check that the number of discounts and dates are the same
        self.T = DayCountConversion(
            self.reference, discount_dates, day_count_convention
        )  # convert dates into fixings (yrf)
        if self.T[0] == 0:
            self.T = self.T[1:]
            discounts = discounts[1:]
        r_zero = (-1.0 / self.T) * log(discounts)
        r_instant = np.append(
            r_zero[0],
            (1.0 / (self.T[:-1] - self.T[1:])) * log(discounts[1:] / discounts[:-1]),
        )
        self.R = interp1d(
            self.T, r_zero, fill_value="extrapolate"
        )  # zero interest rate from 0 to T1 (linear interpolation)
        self.r_t = interp1d(
            self.T, r_instant, kind="next", fill_value="extrapolate"
        )  # instant interest rate (piecewise constant)
        # (needed for target volatility strategy simulation)

    def curve(self, date: float or np.ndarray) -> float or np.ndarray:
        """
        Return the discount factor at a given date expressed in yrf.
        :param date (float or np.ndarray[float]): date expressed in yrf
        :return (float or np.ndarray[float]): discount factor value at the given date
        """
        return exp(-self.R(date) * date)


class EquityForwardCurve(Curve):
    """Equity Forward Curve"""

    def __init__(
        self,
        reference: int or float,
        spot: float,
        discounting_curve: DiscountingCurve,
        repo_rates: np.ndarray = None,
        repo_dates: np.ndarray = None,
        day_count_convention: str = None,
        asset_name: str = None,
    ) -> None:
        """
        Equity Forward Curve class.
        The equity forward curve computes the forward value for a given equity asset at a given date.
        :param reference (int or float): reference date (in days or yrf)
        :param spot (float): spot price of the equity asset
        :param discounting_curve (DiscountingCurve): market discounting curve
        :param repo_rates (np.ndarray[float], default=None): hedging costs for the equity asset in terms of zero repo rates (mu in the paper notation).
        If None then the repo_rates are assumed to be zero. The forward curve builds from the repo_rates the instant hedging costs assuming them pieceswise constant right open.
        :param repo_dates (np.ndarray[int] or np.ndarray[float], default=None): dates of the hedging costs (in days or yrf). If None then the repo_rates are assumed to be zero.
        :param day_count_convention (str, default=None): day count convention (available ACT_365, ACT_360 or None). If None then the repo_dates and reference are assumed to be in yrf.
        :param asset_name (str, default=None): name of the equity asset
        """
        assert type(discounting_curve) == DiscountingCurve
        if repo_rates is None and repo_dates:
            repo_rates = np.array([0.0, 0.0])
            repo_dates = np.array([0.0 + reference, 10.0 + reference])
            day_count_convention = None  # if repo_rates is None then the repo_dates are assumed to be in yrf
        elif (repo_rates is not None and repo_dates is None) or (
            repo_rates is None and repo_dates is not None
        ):
            raise ValueError(
                "repo_rates and repo_dates must be both None or both not None"
            )
        assert len(repo_rates) == len(repo_dates)
        assert (
            repo_rates.ndim == repo_dates.ndim == 1
        )  # repo_rates and repo_dates must be 1D arrays
        assert spot > 0.0  # spot price must be positive
        self.asset_name = asset_name
        self.spot = spot
        self.reference = reference
        self.discounting_curve = discounting_curve
        self.T = DayCountConversion(
            self.reference, repo_dates, day_count_convention
        )  # convert dates into fixings (yrf)
        if self.T[0] != 0.0:  # the
            self.T = np.append(0.0, self.T[:-1])
        self.q_values = np.array([repo_rates[0]])  # array of the instant hedging costs
        for i in range(1, len(self.T)):
            instant_repo_rate = (
                self.T[i] * repo_rates[i] - self.T[i - 1] * repo_rates[i - 1]
            ) / (self.T[i] - self.T[i - 1])
            self.q_values = np.append(self.q_values, instant_repo_rate)
        self.q = interp1d(
            self.T,
            self.q_values,
            kind="previous",
            fill_value="extrapolate",
            assume_sorted=False,
        )  # instant hedging costs (piecewise constant right open)

    def curve(self, date: float or np.ndarray) -> float or np.ndarray:
        """
        Return the forward value at a given date expressed in yrf.
        :param date (float or np.ndarray[float]): date expressed in yrf
        :return (float or np.ndarray[float]): forward price at the given date
        """
        date = np.array(date)
        if date.shape != ():
            return np.asarray(
                [
                    (self.spot / self.discounting_curve(extreme))
                    * exp(-quad_piecewise(self.q, self.T, 0, extreme))
                    for extreme in date
                ]
            )
        else:
            return (self.spot / self.discounting_curve(date)) * exp(
                -quad_piecewise(self.q, self.T, 0, date)
            )


class ForwardVariance(Curve):
    """Forward Variance Curve"""

    def __init__(
        self,
        reference: int or float,
        market_volatility_matrix: np.ndarray,
        strikes: np.ndarray,
        maturity_dates: np.ndarray,
        strike_interp: float or EquityForwardCurve,
        day_count_convention: str = None,
        asset_name: str = None,
    ) -> None:
        """
        Forward Variance Curve.
        The forward variance curve returns the squared of the instantaneous volatility built from a terms of market volatilities
        interpolated on strikes according to strike_interp. The time interpolation will be piecewise constant (right open) on instantaneous forward volatilities.
        :param reference (int or float): reference date (in days or yrf)
        :param market_volatility_matrix (np.ndarray[float]): market volatilities. The shape of the array must be (n_maturity_dates, n_strikes)
        :param strikes (np.ndarray[float]): strikes of the implied volatility surface
        :param maturity_dates (np.ndarray[int] or np.ndarray[float]): maturity dates of the implied volatility surface (in days or yrf)
        :param strike_interp (float or EquityForwardCurve): interpolation rule for the strikes (ATM spot if type(strike_interp)=float or ATM forward if type(strike_interp)=EquityForwardCurve)
        :param day_count_convention (str, default=None): day count convention (available ACT_365, ACT_360 or None). If None then the maturity_dates and reference are assumed to be in yrf.
        :param asset_name (str, default=None): name of the asset
        """
        assert market_volatility_matrix.ndim == 2  # spot_volatility must be 2D array
        assert (
            strikes.ndim == maturity_dates.ndim == 1
        )  # strikes and maturity_dates must be 1D arrays
        assert len(market_volatility_matrix.T) == len(strikes)
        assert len(market_volatility_matrix) == len(maturity_dates)
        assert type(strike_interp) in [
            float,
            EquityForwardCurve,
            np.float64,
            np.float32,
        ]
        self.asset_name = asset_name
        self.reference = reference
        self.market_volatility = market_volatility_matrix
        self.T = DayCountConversion(
            self.reference, maturity_dates, day_count_convention
        )  # convert dates into fixings (yrf)
        if isinstance(strike_interp, EquityForwardCurve):
            # Interpolation with the ATM forward to get the spot volatilities
            self.spot_vol = np.array([])
            self.K = strikes
            matrix_interpolated = interp1d(strikes, market_volatility_matrix, axis=1)(
                strike_interp(self.T)
            )
            for i in range(len(maturity_dates)):
                self.spot_vol = np.append(self.spot_vol, matrix_interpolated[i, i])
        else:
            # Interpolation with the ATM spot to get the spot volatilities
            self.spot_vol = interp1d(strikes, market_volatility_matrix, axis=1)(
                strike_interp
            )

        self.forward_vol = np.array(
            [self.spot_vol[0]]
        )  # forward (instantaneous) volatility from 0 to T1
        for i in range(1, len(self.T)):
            forward_volatility = (
                self.T[i] * (self.spot_vol[i] ** 2)
                - self.T[i - 1] * (self.spot_vol[i - 1] ** 2)
            ) / (self.T[i] - self.T[i - 1])
            self.forward_vol = np.append(self.forward_vol, sqrt(forward_volatility))
        if self.T[0] != 0:
            self.T = np.insert(self.T[:-1], 0, 0)
        # interpolation (piecewise constant right open) of the instantaneous forward volatilities
        self.vol_t = interp1d(
            self.T,
            self.forward_vol,
            kind="previous",
            fill_value="extrapolate",
            assume_sorted=False,
        )

    def curve(self, date: float or np.ndarray) -> float or np.ndarray:
        """
        Return the squared instantaneous volatility at a given date expressed in yrf.
        :param date (float or np.ndarray[float]): date expressed in yrf
        :return (float or np.ndarray[float]): squared instantaneous volatility value at the given date
        """
        return self.vol_t(date) ** 2


class LocalVolatilityCurve:
    """Local Volatility Curve."""

    def __init__(
        self,
        local_volatility_parameters: np.ndarray,
        moneyness_matrix: np.ndarray,
        maturities: np.array,
        asset_name: str = None,
        log_money_interpolation_rule: str = "kruger",
    ):
        """
        Local Volatility Curve.
        The local volatility curve returns the local volatility at a given maturity and log-moneyness. The interpolation of the local volatility curve
        is piecewise constant in time direction and log_money_interpolation_rule in log-moneyness direction. The extrapolation rule is constant both in time and log-moneyness directions.
        :param local_volatility_parameters (np.ndarray[float]): local volatility parameters. The shape of the array is (n_moneyness, n_maturities)
        :param moneyness_matrix (np.ndarray[float]): moneyness matrix. The shape of the array is (n_strikes, n_maturities)
        :param maturities (np.ndarray[float]): maturities expressed in yrf
        :param asset_name (str, default=None): name of the asset
        :param strike_interpolation_rule (str, default="kruger"): interpolation rule for the log-moneyness. Available options are "kruger" and "piecewise"
        """
        assert (
            local_volatility_parameters.ndim == moneyness_matrix.ndim == 2
        )  # local_volatility_parameters and moneyness_matrix must be 2D arrays
        assert len(local_volatility_parameters) == len(moneyness_matrix)
        assert len(local_volatility_parameters.T) == len(maturities)
        assert len(local_volatility_parameters.T) == len(moneyness_matrix.T)
        self.asset_name = asset_name
        self.lv_pars = local_volatility_parameters
        self.log_moneyness = np.log(moneyness_matrix)
        self.T = maturities
        n_dates = len(
            maturities
        )  # it is fundamental this transformation for the piecewise interpolation
        time_idx = tuple(range(n_dates))
        if n_dates > 1:
            self.time_interpolator = interp1d(
                maturities, time_idx, kind="next", fill_value="extrapolate"
            )
        else:
            self.time_interpolator = lambda t: 0

        self.interpolator_strikes = []
        self.money_max = np.array([])
        self.money_min = np.array([])
        self.LV_max = np.array([])
        self.LV_min = np.array([])
        self.time_indexes_of_simulation_grid = None
        if log_money_interpolation_rule == "kruger":
            self.use_kruger = True
        elif log_money_interpolation_rule == "piecewise":
            self.use_kruger = False
        else:
            raise ValueError("strike interpolation rule type not recognized")

    def interpolate_on_time(self, simulation_time_grid: np.ndarray) -> None:
        """
        Interpolate the local volatility curve on a given simulation time grid.
        This method is used to speed up the local volatility simulation, since
        once the simulation time grid is known, the local volatility curve can be
        interpolated on the simulation time grid only once.
        :param time_grid (np.ndarray[float]): simulation time grid expressed in yrf
        """
        assert len(simulation_time_grid) > 1
        self.time_indexes_of_simulation_grid = self.time_interpolator(
            simulation_time_grid
        ).astype(int)
        self.interpolator_strikes = []
        self.money_max = np.array([])
        self.money_min = np.array([])
        self.LV_max = np.array([])
        self.LV_min = np.array([])
        for i in range(self.time_indexes_of_simulation_grid[-1] + 1):
            this_money = self.log_moneyness[:, i]
            this_lv = self.lv_pars[:, i]
            if self.use_kruger:
                self.interpolator_strikes.append(PchipInterpolator(this_money, this_lv))
            else:
                self.interpolator_strikes.append(
                    interp1d(
                        this_money, this_lv, kind="nearest", fill_value="extrapolate"
                    )
                )
            self.money_max = np.append(self.money_max, np.max(this_money))
            self.money_min = np.append(self.money_min, np.min(this_money))
            self.LV_min = np.append(self.LV_min, this_lv[0])
            self.LV_max = np.append(self.LV_max, this_lv[-1])

    def get_time_index_grid(self) -> np.ndarray:
        """
        Get the time index grid of the simulation grid.
        :return (np.ndarray[float]): time index grid
        """
        if self.time_indexes_of_simulation_grid is None:
            raise ValueError(
                "The simulation time grid has not been interpolated yet. Use interpolate_on_time method."
            )
        return self.time_indexes_of_simulation_grid

    def vectorized_call(
        self, index: int, log_k: float or np.ndarray
    ) -> float or np.ndarray:
        """
        Get the local volatility at a given time index and a given log-moneyness array.
        :param index (int): time index. This has been evaluated with the interpolate_on_time method.
        :param log_k (float or np.ndarray): log-moneyness
        :return (float or np.ndarray[float]): local volatility parameters. The shape is the same as log_k
        """
        eta = self.interpolator_strikes[index](log_k)
        eta[log_k < self.money_min[index]] = self.LV_min[index]
        eta[log_k > self.money_max[index]] = self.LV_max[index]
        return eta

    def __call__(self, t: float, log_k: float) -> float:
        """
        Returns the local volatility at a given time and log-moneyness.
        :param t (float): time expressed in yrf
        :param log_k (float): log-moneyness
        :return (float): local volatility parameter
        """
        idx = int(self.time_interpolator(t))
        this_money = self.log_moneyness[:, idx]
        this_lv = self.lv_pars[:, idx]
        if self.use_kruger:
            eta = PchipInterpolator(this_money, this_lv)(log_k)
        else:
            eta = interp1d(
                this_money, this_lv, kind="nearest", fill_value="extrapolate"
            )(log_k)
        eta[log_k < this_money[0]] = this_lv[
            0
        ]  # extrapolation rule (constant in log-moneyness)
        eta[log_k > this_money[-1]] = this_lv[
            -1
        ]  # extrapolation rule (constant in log-moneyness)
        return eta


class Black:
    """Black and Scholes model"""

    def __init__(
        self,
        fixings: np.ndarray,
        forward_curve: EquityForwardCurve or list,
        variance_curve: ForwardVariance or list,
        correlation_matrix: np.ndarray = None,
        sampling: str = "standard",
    ) -> None:
        """
        Black and Scholes model.
        The class implements both the single asset and multi-asset Black and Scholes models depending on the input.
        If the forward_curve and the variance_curve are lists, the model is multi-asset, otherwise it is single asset.
        :param fixings (np.ndarray[float]): array of fixings dates (in yrf) at which the model will be evaluated
        :param forward_curve (EquityForwardCurve or list[EquityForwardCurve]): forward curve(s) of the underlying asset(s)
        :param variance_curve (ForwardVariance or list[ForwardVariance]): variance curve(s) of the underlying asset(s)
        :param correlation_matrix (np.ndarray[float], default=None): correlation matrix of the underlying assets. This parameter is read
        only if the model is multi-asset. In this case, if it is not provided (None), the correlation matrix is assumed to be the identity matrix.
        :param sampling (str, default="standard"): sampling method used to simulate the model. The options are: "standard" (default), "antithetic".
        The last method is a control variate technique to improve the accuracy of the Monte Carlo simulation.
        """
        assert type(fixings) == np.ndarray
        assert fixings.ndim == 1
        self.n_fixings = len(fixings)
        if type(forward_curve) == list and len(forward_curve) > 1:
            # N dimensional Black and Scholes model
            assert all(isinstance(x, EquityForwardCurve) for x in forward_curve)
            assert type(variance_curve) == list
            assert all(isinstance(x, ForwardVariance) for x in variance_curve)
            assert len(variance_curve) == len(forward_curve)
            self.n_stocks = len(forward_curve)
            if (
                correlation_matrix is None
            ):  # if no correlation matrix is provided, we assume no correlated assets
                self.cholesky_matrix = np.eye(self.n_stocks)
            else:
                assert type(correlation_matrix) == np.ndarray
                assert correlation_matrix.shape == (self.n_stocks, self.n_stocks)
                self.cholesky_matrix = np.linalg.cholesky(correlation_matrix)
            self.forward_values = np.zeros((self.n_fixings, self.n_stocks))
            self.variance_values = np.zeros((self.n_stocks, self.n_fixings))
            for i in range(self.n_stocks):
                self.forward_values[:, i] = forward_curve[i](fixings)
                for j in range(self.n_fixings):
                    if j == 0:
                        self.variance_values[i, j] = quad_piecewise(
                            variance_curve[i], variance_curve[i].T, 0.0, fixings[j]
                        )  # variance from t=0 to t=fixings[j]
                    else:
                        self.variance_values[i, j] = quad_piecewise(
                            variance_curve[i],
                            variance_curve[i].T,
                            fixings[j - 1],
                            fixings[j],
                        )  # variance from t=fixings[j-1] to t=fixings[j]
        else:
            # Single asset Black and Scholes model
            self.n_stocks = 1
            assert type(forward_curve) == EquityForwardCurve
            assert type(variance_curve) == ForwardVariance
            self.forward_values = forward_curve(fixings)
            self.variance_values = np.zeros(self.n_fixings)
            for j in range(self.n_fixings):
                if j == 0:
                    self.variance_values[j] = quad_piecewise(
                        variance_curve, variance_curve.T, 0.0, fixings[j]
                    )
                else:
                    self.variance_values[j] = quad_piecewise(
                        variance_curve, variance_curve.T, fixings[j - 1], fixings[j]
                    )
        if sampling == "standard":
            self.antithetic_sampling = False
        elif sampling == "antithetic":
            self.antithetic_sampling = True
        else:
            raise ValueError(
                "Sampling type not recognized. Please use 'standard' or 'antithetic'."
            )

    def simulate(
        self,
        random_generator: np.random = None,
        n_sim: int or float = 1,
        seed: int = 14,
        return_log_martingale: bool = False,
    ) -> np.ndarray:
        """
        Monte Carlo simulation of the Black and Scholes model. The method returns the simulated paths of the underlying asset(s) at the fixing dates
        provided at the construction of the model. The returned array has shape (n_sim, n_fixings, n_stocks) for the multi-asset model, while
        (n_sim, n_fixings) for the single-asset one.
        :param random_generator (np.random, default=None): random number generator. If not provided, then then default one is used with seed as seed value for the generation.
        :param n_sim (int or float, default=1): number of Monte Carlo samplings. If float value is provided, then it is casted to int. If the sampling method is antithetic, then the number of simulations is doubled:
        n_sim = 2 * n_sim (the first half of the simulations are the standard ones, the second half are the antithetic ones).
        :param seed (int, default=14): seed value for the default random number generator (when random_generator is None).
        :param return_log_martingale (bool, default=False): if True, the method returns the simulated log martingale at the fixing dates.
        :return simulated_paths (np.ndarray[float]): array containing the simulated paths (or log martingales if return_log_martingale is True) of the underlying asset(s) at the fixing dates.
        The array shape is (n_sim, n_fixings, n_stocks) for the multi-asset model, while (n_sim, n_fixings) for the single-asset one.
        """
        assert n_sim > 0  # number of simulations must be positive
        n_mc_sim = int(2 * n_sim) if self.antithetic_sampling else int(n_sim)
        n_sim = int(n_sim)
        if random_generator is None:
            random_generator = np.random
            random_generator.seed(seed)
        if self.n_stocks == 1:
            logmartingale = np.zeros((n_mc_sim, self.n_fixings))
            for i in range(self.n_fixings):
                Z = random_generator.randn(n_sim)
                if self.antithetic_sampling:
                    Z = np.concatenate((Z, -Z))  # attach the antithetic samples
                if i == 0:
                    logmartingale[:, i] = (
                        -0.5 * self.variance_values[i]
                        + sqrt(self.variance_values[i]) * Z
                    )
                else:
                    logmartingale[:, i] = (
                        logmartingale[:, i - 1]
                        - 0.5 * self.variance_values[i]
                        + sqrt(self.variance_values[i]) * Z
                    )
            if return_log_martingale:
                return logmartingale
            else:
                return exp(logmartingale) * self.forward_values
        else:
            logmartingale = np.zeros((n_mc_sim, self.n_fixings, self.n_stocks))
            for i in range(self.n_fixings):
                Z = random_generator.randn(n_sim, self.n_stocks)
                if self.antithetic_sampling:
                    Z = np.concatenate((Z, -Z))  # attach the antithetic samples
                ep = self.cholesky_matrix @ Z.T  # matrix of correlated random variables
                for j in range(self.n_stocks):
                    if i == 0:
                        logmartingale[:, i, j] = (
                            -0.5 * self.variance_values[j, i]
                            + sqrt(self.variance_values[j, i]) * ep[j]
                        )
                    elif i != 0:
                        logmartingale[:, i, j] = (
                            logmartingale[:, i - 1, j]
                            - 0.5 * self.variance_values[j, i]
                            + sqrt(self.variance_values[j, i]) * ep[j]
                        )
            if return_log_martingale:
                return logmartingale
            else:
                return exp(logmartingale) * self.forward_values


class LV_model:
    """Local Volatility Model"""

    def __init__(
        self,
        fixings: np.array,
        forward_curve: EquityForwardCurve or list,
        local_vol_curve: LocalVolatilityCurve or list,
        n_euler_grid: int = 100,
        correlation_matrix: np.ndarray = None,
        sampling: str = "standard",
        return_grid_values_for_tvs: bool = False,
    ) -> None:
        """
        Local Volatility Model.
        The class implements both the single asset and multi-asset Black and Scholes models depending on the input.
        If the forward_curve and the local_vol_curve are lists, the model is multi-asset, otherwise it is single asset.
        :param fixings (np.array[float]): array of fixings dates (in yrf) at which the model will be evaluated
        :param forward_curve (EquityForwardCurve or list[EquityForwardCurve]): forward curve(s) of the underlying asset(s)
        :param variance_curve (LocalVolatilityCurve or list[LocalVolatilityCurve]): variance curve(s) of the underlying asset(s)
        :param n_euler_grid (int, default=100): number of points in the Euler grid that is built to discretize the simulation from fixings[i] to fixings[i+1]
        :param correlation_matrix (np.ndarray[float], default=None): correlation matrix of the underlying assets. This parameter is read
        only if the model is multi-asset. In this case, if it is not provided (None), the correlation matrix is assumed to be the identity matrix.
        :param sampling (str, default="standard"): sampling method used to simulate the model. The options are: "standard" (default), "antithetic".
        :param return_grid_values_for_tvs (bool, default=False): if True, the simulated method returns a tuple containing two np.ndarray: the spot prices and the instantaneous
        volatilities evaluated on the Euler grid (and not at the fixing dates). The two arrays have size (n_mc_sim, n_euler_grid * n_fixings, n_stocks).
        This option is read only if the model is multi-asset.
        """
        assert type(fixings) == np.ndarray
        assert fixings.ndim == 1
        if fixings[0] == 0.0:  # skip the reference date
            raise ValueError("Fixings must not start at 0.0")
        self.local_vol = local_vol_curve
        self.euler_time_grid, self.dt = get_euler_grid(
            fixings, n_euler_grid
        )  # get the Euler grid for the model simulation and the time step of each
        assert self.euler_time_grid.ndim == self.dt.ndim == 1
        assert self.euler_time_grid.shape[0] == n_euler_grid * len(fixings)
        assert len(self.dt) == len(fixings)
        self.sqrt_dt = np.sqrt(
            self.dt
        )  # squared time interval (needed for Euler Scheme) (it can be computed just one time in the constructor)
        self.N_grid = n_euler_grid
        self.n_fixings = len(fixings)
        self.n_time_grid = len(self.euler_time_grid)
        if type(forward_curve) == list and len(forward_curve) > 1:
            # Multi-asset LV model
            assert all(isinstance(x, EquityForwardCurve) for x in forward_curve)
            assert type(local_vol_curve) == list
            assert len(local_vol_curve) == len(forward_curve)
            assert all(isinstance(x, LocalVolatilityCurve) for x in local_vol_curve)
            self.n_stocks = len(forward_curve)
            if (
                correlation_matrix is None
            ):  # if no correlation matrix is provided, we assume no correlated assets
                self.cholesky_matrix = np.eye(self.n_stocks)
            else:
                assert type(correlation_matrix) == np.ndarray
                assert correlation_matrix.shape == (self.n_stocks, self.n_stocks)
                self.cholesky_matrix = np.linalg.cholesky(correlation_matrix)
            self.return_grid_values_for_tvs = return_grid_values_for_tvs  # the model returns the state (stock value and volatility for the target_vol dynamics)
            if self.return_grid_values_for_tvs:
                self.forward_values = np.zeros((self.n_time_grid, self.n_stocks))
            else:
                self.forward_values = np.zeros((self.n_fixings, self.n_stocks))
            self.time_indexes_matrix = np.array([])
            for i in range(self.n_stocks):
                if self.return_grid_values_for_tvs:
                    self.forward_values[:, i] = forward_curve[i](self.euler_time_grid)
                else:
                    self.forward_values[:, i] = forward_curve[i](fixings)
                self.local_vol[i].interpolate_on_time(
                    self.euler_time_grid
                )  # interpolate the local volatility curve on the Euler grid
                time_indexes = self.local_vol[
                    i
                ].get_time_index_grid()  # get the time indexes of the local volatility curve used to interpolate on the Euler grid
                if i == 0:
                    self.time_indexes_matrix = time_indexes
                else:
                    self.time_indexes_matrix = np.vstack(
                        [self.time_indexes_matrix, time_indexes]
                    )
        else:
            # Single asset LV model"""
            assert type(forward_curve) == EquityForwardCurve
            assert type(local_vol_curve) == LocalVolatilityCurve
            self.n_stocks = 1
            self.forward_values = forward_curve(fixings)
            self.local_vol.interpolate_on_time(
                self.euler_time_grid
            )  # interpolate the local volatility curve on the Euler grid
            self.time_indexes = (
                self.local_vol.get_time_index_grid()
            )  # get the time indexes of the local volatility curve used to interpolate on the Euler grid
            if return_grid_values_for_tvs:
                raise ValueError(
                    "return_grid_values_for_tvs is not available for the single-asset local volatility model."
                )

        if sampling == "standard":
            self.antithetic_sampling = False
        elif sampling == "antithetic":
            self.antithetic_sampling = True
        else:
            raise ValueError(
                "Sampling type not recognized. Please use 'standard' or 'antithetic'."
            )

    def simulate(
        self, random_generator: np.random = None, n_sim: int = 1, seed: int = 14
    ) -> np.ndarray or tuple:
        """
        Monte Carlo simulation of the Local Volatility model. The method returns the simulated paths of the underlying asset(s) at the fixing dates
        provided at the construction of the model. The returned array has shape (n_sim, n_fixings, n_stocks) for the multi-asset model, while
        (n_sim, n_fixings) for the single-asset one. If return_grid_values_for_tvs is True and the model is multi asset, then the method returns
        a tuple containing two np.ndarray: the spot prices and the instantaneous volatilities evaluated on the whole Euler grid.
        The two arrays have size (n_mc_sim, n_euler_grid * n_fixings, n_stocks).
        :param random_generator (np.random, default=None): random number generator. If not provided, then then default one is used with seed as seed value for the generation.
        :param n_sim (int or float, default=1): number of Monte Carlo samplings. If float value is provided, then it is casted to int. If the sampling method is antithetic, then the number of simulations is doubled:
        n_sim = 2 * n_sim (the first half of the simulations are the standard ones, the second half are the antithetic ones).
        :param seed (int, default=14): seed value for the default random number generator (when random_generator is None).
        :return simulated_paths (np.ndarray[float] or tuple): array containing the simulated paths of the underlying asset(s) at the fixing dates.
        The array shape is (n_sim, n_fixings, n_stocks) for the multi-asset model, while (n_sim, n_fixings) for the single-asset one.
        """
        assert n_sim > 0  # number of simulations must be positive
        n_mc_sim = int(2 * n_sim) if self.antithetic_sampling else int(n_sim)
        n_sim = int(n_sim)
        if random_generator is None:
            random_generator = np.random
            random_generator.seed(seed)
        time_index = 0  # index of the fixing grid
        dt = self.dt[time_index]
        dt_root_squared = self.sqrt_dt[time_index]
        # Single Asset simulation
        if self.n_stocks == 1:
            log_martingale_at_fixing_dates = np.zeros((n_mc_sim, self.n_fixings))
            for i in range(self.n_time_grid):
                Z = random_generator.randn(n_sim)
                if self.antithetic_sampling:  # Build antithetic variables if needed
                    Z = np.concatenate((Z, -Z))
                # Euler scheme
                if i == 0:
                    vol = self.local_vol.vectorized_call(0, 0.0)
                    log_X_t = -0.5 * dt * (vol**2) + vol * dt_root_squared * Z
                elif i != 0:
                    vol = self.local_vol.vectorized_call(
                        self.time_indexes[i - 1], log_X_t
                    )
                    log_X_t = (
                        log_X_t - 0.5 * dt * (vol**2) + vol * dt_root_squared * Z
                    )
                counter = i + 1

                # when reached the evaluation fixing, then save the results in the report variable
                if (counter % self.N_grid) == 0:
                    log_martingale_at_fixing_dates[:, time_index] = log_X_t
                    time_index += 1
                    if counter < self.n_time_grid:
                        dt = self.dt[time_index]
                        dt_root_squared = self.sqrt_dt[time_index]
            return exp(log_martingale_at_fixing_dates) * self.forward_values
        # Multi Asset simulation
        else:
            if not self.return_grid_values_for_tvs:
                log_martingale_at_fixing_dates = np.zeros(
                    (n_mc_sim, self.n_fixings, self.n_stocks)
                )
            else:
                instant_log_martingale = np.zeros(
                    (n_mc_sim, self.n_time_grid, self.n_stocks)
                )
                instant_vola = np.zeros((n_mc_sim, self.n_time_grid, self.n_stocks))

            log_X_t = np.zeros((n_mc_sim, self.n_stocks))
            vol = np.zeros((n_mc_sim, self.n_stocks))
            for i in range(self.n_time_grid):
                Z = np.random.randn(n_sim, self.n_stocks)
                if self.antithetic_sampling:  # Build antithetic variables if needed
                    Z = np.concatenate((Z, -Z))
                ep = self.cholesky_matrix @ Z.T  # matrix of correlated random variables
                for j in range(self.n_stocks):
                    # Euler scheme
                    if i == 0:
                        vol[:, j] = self.local_vol[j].vectorized_call(0, log_X_t[:, j])
                        log_X_t[:, j] = (
                            -0.5 * dt * (vol[:, j] ** 2)
                            + vol[:, j] * dt_root_squared * ep[j]
                        )
                    elif i != 0:
                        vol[:, j] = self.local_vol[j].vectorized_call(
                            self.time_indexes_matrix[j, i - 1], log_X_t[:, j]
                        )
                        log_X_t[:, j] = (
                            log_X_t[:, j]
                            - 0.5 * dt * (vol[:, j] ** 2)
                            + vol[:, j] * dt_root_squared * ep[j]
                        )

                if (
                    self.return_grid_values_for_tvs
                ):  # save the values of the martingale at the Euler grid fixing dates
                    instant_log_martingale[:, i, :] = log_X_t
                    instant_vola[:, i, :] = vol
                counter = i + 1
                # when reached the evaluation fixing, then save the results in the report variable
                if (counter % self.N_grid) == 0:
                    if not self.return_grid_values_for_tvs:
                        log_martingale_at_fixing_dates[:, time_index, :] = log_X_t
                    time_index += 1
                    if counter < self.n_time_grid:
                        dt = self.dt[time_index]
                        dt_root_squared = self.sqrt_dt[time_index]

            if self.return_grid_values_for_tvs:
                return exp(instant_log_martingale) * self.forward_values, instant_vola
            else:
                return exp(log_martingale_at_fixing_dates) * self.forward_values


def Vanilla_PayOff(
    simulated_paths: np.ndarray,
    strike: float,
    call_put: float = 1.0,
    sampling: str = "standard",
) -> np.ndarray:
    """
    Payoff function of a plain vanilla.
    :param simulated_paths (np.ndarray[float]): random variable
    :param strike (float): strike of the option
    :param call_put (float, default=1.0) : 1 for call, -1 for put
    :param sampling (str, default="standard"): sampling method used to simulate the model. The options are: "standard" (default), "antithetic".
    If the sampling is "antithetic", the first half of the simulated_paths contains the standard samples, while the second half contains the antithetic samples.
    :return (np.ndarray[float]): payoff of the vanilla option with size simulated_paths.shape[0] or simulated_paths.shape[0]/2 if antithetic sampling is used.
    """
    zero = np.zeros(simulated_paths.shape)
    pay = np.maximum(zero, call_put * (simulated_paths - strike))
    if sampling == "standard":
        return np.maximum(pay, zero)
    elif sampling == "antithetic":
        pay1, pay2 = np.split(np.maximum(pay, zero), 2)  # for antithetic sampling
        return 0.5 * (pay1 + pay2)
    else:
        raise ValueError(
            "Sampling method not recognized (available methods: standard, antithetic)"
        )
