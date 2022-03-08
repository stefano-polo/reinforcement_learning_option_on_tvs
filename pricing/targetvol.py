import numpy as np
from numpy import exp, sqrt
from pricing import Curve, ForwardVariance, EquityForwardCurve, DiscountingCurve
from pricing_utility import quad_piecewise
from numpy.linalg import cholesky
from scipy.interpolate import interp1d
from scipy.optimize import minimize, LinearConstraint, Bounds

"""
Black and Scholes model for target volatility pricing.
The notation is based on the paper in the tex directory
"""

class Drift(Curve):
    """Time dependent repo rates curve"""

    def __init__(self, forward_curves: list) -> None:
        """
        Drift curve. This function builds the mu function presented in the paper.
        The class returns (with the curve method) the instantaneous repo rates of the asset contained in the
        basket at a given time. The drift curve is a piecewise constant function (right open) of time (the first element
        of the time-grid definition must be 0.0).
        :param forward_curves [list[EquityCurve]]: list of equity curves in the basket.
        """
        assert type(forward_curves) == list and all(isinstance(x, EquityForwardCurve) for x in forward_curves)
        assert len(forward_curves) > 1
        # time grid of the piecewise constant function defined as union of the time grids of the equity curves in the basket
        self.T = time_grid_union(curve_array_list=forward_curves)
        n_dim = len(forward_curves)
        self.mu = forward_curves[0].q(self.T)
        self.mu = np.stack((self.mu, forward_curves[1].q(self.T)), axis=1)
        """Calculating the instant repo rates"""
        for i in range(2, n_dim):
            self.mu = np.insert(self.mu, len(self.mu.T), forward_curves[i].q(self.T), axis=1)
        self.m = interp1d(self.T, self.mu, axis=0, kind='previous', fill_value="extrapolate", assume_sorted=False)

    def curve(self, date: float or np.ndarray) -> np.ndarray:
        """
        Returns the instantaneous repo rates of the asset contained in the basket at a given time.
        :param date (float or np.ndarray[float]): time expressed in yrf.
        :return (np.ndarray[float]): instantaneous repo rates of the asset contained in the basket at a given time. The
        array shape is (n_dates, n_stocks).
        """
        return self.m(date)


class CholeskyTDependent(Curve):
    """Time dependent cholesky variance-covariance matrix"""

    def __init__(self, variance_curves: list, correlation_chole: np.ndarray = None) -> None:
        """
        Cholesky Time Dependent Curve. This function builds the cholesky matrix presented in the paper with nu notation.
        The nu matrix is the cholesky matrix of the covariance matrix of the basket at a given time.
        The cholesky curve is a piecewise constant function (right open) of time (the first element of the time-grid definition must be 0.0).
        :param variance_curves (list[ForwardVariance]): list of forward variance curves in the basket.
        :param correlation_chole (np.ndarray[float], default=None): choleksy decomposition of the correlation matrix of the basket.
        If None, the assets are assumed to be independent.
        """
        assert type(variance_curves) == list and all(isinstance(x, ForwardVariance) for x in variance_curves)
        assert len(variance_curves) > 1
        if correlation_chole is None:  # if the correlation is not provided then assume non-correlated assets
            correlation_chole = np.eye(len(correlation_chole))
        # time grid of the piecewise constant function defined as union of the time grids of the equity curves in the basket
        self.T = time_grid_union(curve_array_list=variance_curves)
        n_times = len(self.T)
        n_dim = len(variance_curves)
        self.nu = np.zeros((n_dim, n_dim, n_times))
        identity_matrix = np.identity(n_dim)
        for i in range(n_times):
            vol = np.zeros(n_dim)
            for j in range(n_dim):
                vol[j] = sqrt(variance_curves[j](self.T[i]))
            vol = identity_matrix * vol
            self.nu[:, :, i] = vol @ correlation_chole
        self.n = interp1d(self.T, self.nu, axis=2, kind='previous', fill_value="extrapolate", assume_sorted=False)

    def curve(self, date: float or np.ndarray) -> np.ndarray:
        """
        Returns the cholesky matrix of the covariance matrix of the basket at a given time.
        :param date (float or np.ndarray[float]): time expressed in yrf.
        :return (np.ndarray[float]): cholesky matrix of the covariance matrix of the basket at a given time.
        The array shape is (n_stocks, n_stocks, n_dates).
        """
        return self.n(date)


class Strategy:
    """Time dependent allocation strategy (alpha in the paper notation)"""

    def __init__(self, strategy: np.ndarray = None, dates: np.ndarray = None) -> None:
        """
        Time dependent allocation strategy. This function builds the alpha function presented in the paper.
        The alpha function is a piecewise constant function (right open) of time (the first element of the time-grid definition must be 0.0).
        :param strategy (np.ndarray[float], default=None): array of the allocation strategy. If None, the strategy is not initialized.
        :param dates (np.ndarray[float], default=None): array of the dates. If None, the dates are not initialized.
        """
        self.T = dates
        self.alpha_t = strategy
        if self.alpha_t is not None and self.T is not None:
            if self.T[0] != 0.0:
                raise ValueError("The first date should be 0.0")
            assert len(self.T) == len(self.alpha_t), "The time grid of strategy and dates should have the same length"
            self.a_t = interp1d(self.T, self.alpha_t, axis=0, kind='previous', fill_value="extrapolate", assume_sorted=False)
        elif self.alpha_t is None and self.T is None:
            self.a_t = None
        else:
            raise ValueError("The strategy and dates should be both provided or both not provided")

    def Mark_strategy(self, mu: Drift, nu: CholeskyTDependent) -> None:
        """
        Computes the optimal allocation strategy for the Black and Scholes model, assuming
        free bounds for the strategy. The closed formula implemented is the number (36).
        The function name reflects the fact that the free optimal strategy needs to solve a Markowitz problem.
        :param mu (Drift): drift of the portfolio.
        :param nu (CholeskyTDependent): cholesky matrix of the covariance matrix of the portfolio.
        """
        assert isinstance(mu, Drift)
        assert isinstance(nu, CholeskyTDependent)
        n_dim = len(mu(0))
        self.T = np.union1d(mu.T, nu.T)
        self.alpha_t = np.zeros((len(self.T), n_dim))   # time dependent allocation strategy
        for i in range(len(self.T)):
            a_plus = Markowitz_solution(mu(self.T[i]), nu(self.T[i]), 1)  # compute the optimal allocation for the portfolio through the closed formula
            a_minus = - a_plus
            if loss_function(a_plus, mu(self.T[i]), nu(self.T[i])) > loss_function(a_minus, mu(self.T[i]), nu(self.T[i])):
                self.alpha_t[i] = a_minus
            else:
                self.alpha_t[i] = a_plus
        self.a_t = interp1d(self.T, self.alpha_t, axis=0, kind='previous', fill_value="extrapolate", assume_sorted=False)

    def optimization_constrained(self, mu: Drift, nu: CholeskyTDependent, long_limit=25/100, short_limit=25/100, n_trial=20, seed=13, constraint_strategy=1) -> None:
        """
        Computes the optimal allocation strategy for the Black and Scholes model, assuming some constraints on the allocation strategy.
        :param mu (Drift): drift of the portfolio.
        :param nu (CholeskyTDependent): cholesky matrix of the covariance matrix of the portfolio.
        :param long_limit (float, default=25/100): maximum proportion of the portfolio that can be invested in a long position.
        This parameter is read only for constraint_strategy=[2,3]
        :param short_limit (float, default=25/100): maximum proportion of the portfolio that can be invested in a short position.
        This parameter is read only for constraint_strategy=[2,3]
        :param n_trial (int, default=20): number of trials to find the optimal strategy with different initial random guesses.
        :param seed (int, default=13): seed for the random number generator for the initial random guesses.
        :param constraint_strategy (int, default=1): constraint to impose on the allocation strategy. If constraint is 1,
        the strategy is constrained to be long on all the positions (alpha_i > 0) and the sum of all the allcoation must be = 1
        (sum_i alpha_i = 1). If constraint is 1, then absolute value of each allocation is bounded by the long_limit (|alpha_i| <= long_limit).
        If constraint is 2, then the sum of the long (alpha_i>0) allocation positions must be lower than long_limit, while the sum of the short (alpha_i<0)
        allocation positions must be lower than short_limit.
        """
        assert isinstance(mu, Drift)
        assert isinstance(nu, CholeskyTDependent)
        n_dim = len(mu(0))
        self.T = np.union1d(mu.T, nu.T)
        if np.max(mu.T) > np.max(nu.T):    # check control to avoid denominator divergence
            self.T = self.T[np.where(self.T <= np.max(nu.T))[0]]
        self.alpha_t = np.zeros((len(self.T), n_dim))   # time dependent allocation strategy
        for i in range(len(self.T)):
            if constraint_strategy == 1:
                result = optimization_only_long(mu(self.T[i]), nu(self.T[i]),  n_trial=n_trial, seed=seed, guess=None)
            elif constraint_strategy == 2:
                result = optimization_limit_position(mu(self.T[i]), nu(self.T[i]), limit_position=long_limit, n_trial=n_trial, seed=seed)
            else:
                result = optimization_long_short_position(mu(self.T[i]), nu(self.T[i]), long_limit=long_limit, short_limit=short_limit, n_trial=n_trial, seed=seed)
            self.alpha_t[i] = result
        self.a_t = interp1d(self.T, self.alpha_t, axis=0, kind='previous', fill_value="extrapolate", assume_sorted=False)

    def Intuitive_strategy1(self, forward_curves_list: list, maturity_date: float) -> None:
        """
        Intuitive strategy that consists on investing all on the asset with maximum growth at maturity.
        :param forward_curves_list (list): list of forward curves of the assets in the basket
        :param maturity_date (float): maturity date in yrf
        """
        assert type(forward_curves_list) == list and all(isinstance(x, EquityForwardCurve) for x in forward_curves_list)
        assert len(forward_curves_list) > 1
        asset_index = Max_forward_maturity(forward_curves_list, maturity_date)
        self.T = np.array([0, maturity_date])
        self.alpha_t = np.zeros((2, len(forward_curves_list)))
        self.alpha_t[:, asset_index] = 1
        self.a_t = interp1d(self.T, self.alpha_t, axis=0, kind='previous', fill_value="extrapolate", assume_sorted=False)

    def Intuitive_strategy2(self, mu: Drift) -> None:
        """
        Intuitive strategy that consists on investing all on the asset with minimum mu parameter at each allocation time.
        :param mu (Drift): drift of the portfolio.
        """
        assert isinstance(mu, Drift)
        asset_index = Min_mu_each_time(mu)
        self.T = mu.T
        self.alpha_t = np.zeros((len(asset_index), len(mu(0.))))
        for j in range(len(asset_index)):
            self.alpha_t[j, int(asset_index[j])] = 1
        self.a_t = interp1d(self.T, self.alpha_t, axis=0, kind='previous', fill_value="extrapolate", assume_sorted=False)

    def Intuitive_strategy3(self, mu: Drift, nu: CholeskyTDependent) -> None:
        """
        Intuitive strategy that consists on investing all on the asset with minimum mu/nu variable at each allocation time.
        :param mu (Drift): drift of the portfolio.
        :param nu (CholeskyTDependent): Cholesky decomposition of the covariance matrix of the portfolio.
        """
        assert isinstance(mu, Drift)
        assert isinstance(nu, CholeskyTDependent)
        asset_index, self.T = Min_mu_nu_each_time(mu, nu)
        self.alpha_t = np.zeros((len(asset_index), len(mu(0.))))
        for j in range(len(asset_index)):
            self.alpha_t[j, int(asset_index[j])] = 1
        self.a_t = interp1d(self.T, self.alpha_t, axis=0, kind='previous', fill_value="extrapolate", assume_sorted=False)

    def __call__(self, date: float or np.ndarray) -> np.ndarray:
        """
        Return the allocation strategy at a given date. The value of the allocation strategy
        depends on how the private member self.a_t has been built. The allocation strategy is
        a piecewise constant function of time.
        :param date (float or np.ndarray[float]): date in yrf
        :return (np.ndarray[float]): allocation strategy at date. The shape of the array is (n_dates, n_stocks).
        """
        return self.a_t(date)


class TVSForwardCurve:
    """Forward curve of the target volatility strategy portfolio."""

    def __init__(self, reference: int or float, vola_target: float, spot_price: float, mu: Drift,
                 nu: CholeskyTDependent, discounting_curve: Curve, strategy: Strategy) -> None:
        """
        Target Volatility Strategy Forward Curve.
        :param reference (int or float): reference date in yrf
        :param vola_target (float): target volatility of the portfolio
        :param spot_price (float): spot price of the portfolio
        :param mu (Drift): drift of the assets underlying the portfolio
        :param nu (CholeskyTDependent): Cholesky decomposition of the covariance matrix of assets underlying the portfolio
        :param discounting_curve (Curve): discounting curve of the portfolio
        :param strategy (Strategy): allocation strategy of the risky portfolio.
        """
        assert isinstance(mu, Drift)
        assert isinstance(nu, CholeskyTDependent)
        assert isinstance(discounting_curve, DiscountingCurve)
        assert isinstance(strategy, Strategy) or strategy == None
        self.reference = reference
        self.vol = vola_target     # target volatility
        self.alpha = strategy
        self.I_0 = spot_price
        self.mu = mu
        self.nu = nu
        self.D = discounting_curve
        self.alpha = strategy

    def set_strategy(self, strategy: Strategy) -> None:
        """
        Set the allocation strategy of the risky portfolio.
        :param strategy (Strategy): allocation strategy of the risky portfolio.
        """
        assert isinstance(strategy, Strategy)
        self.alpha = strategy

    def __call__(self, date: float or np.ndarray) -> np.ndarray:
        """
        Return the forward price of the target volatility portfolio  at a given date.
        :param date (float or np.ndarray[float]): date in yrf
        :return (floar or np.ndarray[float]): forward price of the target volatility portfolio at date. The shape of the array is (n_dates).
        """
        if self.alpha is None:
            raise ValueError("Please set a strategy first")
        date = np.array(date)
        function = lambda x: self.vol * np.sum(self.alpha(x) * self.mu(x), axis=1) / np.linalg.norm(np.sum((self.alpha(x).T * (self.nu(x)).transpose(1, 0, 2)), axis=1).T, axis=1)
        if date.shape != ():
            return np.asarray([(self.I_0 / self.D(extreme)) * exp(-quad_piecewise(function, self.alpha.T, self.reference, extreme, vectorized=False)) for extreme in date])
        else:
            return (self.I_0 / self.D(date)) * exp(-quad_piecewise(function, self.alpha.T, self.reference, date, vectorized=False))


class TargetVolatilityStrategy:
    """Target Volatility price process under the Black and Scholes model"""

    def __init__(self, fixings: np.ndarray, forward_curve: TVSForwardCurve, sampling: str = "standard") -> None:
        """
        Target Volatility Strategy under the Black and Scholes model.
        :param fixings (np.ndarray[float]): array of fixings dates (in yrf) at which the model will be evaluated
        :param forward_curve (TVSForwardCurve): forward curve of the target volatility strategy portfolio
        :param sampling (str, default="standard"): sampling method used to simulate the model. The options are: "standard" (default), "antithetic".
        The last method is a control variate technique to improve the accuracy of the Monte Carlo simulation.
        """
        assert isinstance(forward_curve, TVSForwardCurve)
        assert type(fixings) == np.ndarray
        assert fixings.ndim == 1
        self.alpha = forward_curve.alpha
        self.nu = forward_curve.nu
        self.vol = forward_curve.vol      # target volatility
        simulation_dates = fixings
        self.n_times = len(fixings)
        self.n_stocks = int(len(self.nu(0.)))  # number of assets in the risky portfolio
        self.forward_values = forward_curve(simulation_dates)
        self.simulation_dates = simulation_dates
        simulation_dates = np.append(0., simulation_dates)
        self.dt = np.diff(simulation_dates)

        if sampling == "standard":
            self.antithetic_sampling = False
        elif sampling == "antithetic":
            self.antithetic_sampling = True
        else:
            raise ValueError("Sampling type not recognized. Please choose between 'standard' and 'antithetic'.")

    def simulate(self, random_generator: np.random = None, n_sim: int or float = 1, seed: int = 14):
        """
        Monte Carlo simulation of the target volatility strategy portfolio under the Black and Scholes model.
        :param random_generator (np.random, default=None): random number generator. If not provided, then then default one is used with seed as seed value for the generation.
        :param n_sim (int or float, default=1): number of Monte Carlo samplings. If float value is provided, then it is casted to int. If the sampling method is antithetic, then the number of simulations is doubled:
        n_sim = 2 * n_sim (the first half of the simulations are the standard ones, the second half are the antithetic ones).
        :param seed (int, default=14): seed value for the default random number generator (when random_generator is None).
        :return simulated_paths (np.ndarray[float]): array containing the simulated paths of the target volatility strategy at the fixing dates.
        The array shape is (n_sim, n_fixings).
        """
        if random_generator is None:
            random_generator = np.random
            random_generator.seed(seed)
        n_sim = int(n_sim)
        n_mc_sim = int(2 * n_sim) if self.antithetic_sampling else int(n_sim)
        log_I_t = np.zeros((n_mc_sim, self.n_times))
        for i in range(self.n_times):
            Z = random_generator.randn(n_sim, self.n_stocks)
            if self.antithetic_sampling:
                Z = np.concatenate((Z, -Z))
            if i == 0:
                prod = self.alpha(0.) @ self.nu(0.)
            else:
                prod = self.alpha(self.simulation_dates[i-1]) @ self.nu(self.simulation_dates[i-1])
            omega_t = self.vol / np.linalg.norm(prod)
            log_I_t[:, i] = log_I_t[:, i-1] - 0.5 * (self.vol**2) * self.dt[i] + sqrt(self.dt[i]) * ((omega_t * prod) @ Z.T)

        return exp(log_I_t) * self.forward_values


def time_grid_union(curve_array_list: list) -> np.ndarray:
    """
    Returns the union of the time grids of the curves in the list.
    :param curve_array_list (list): list of curves.
    :return time_grid (np.ndarray[float]): union of the time grids of the curves in the list.
    """
    assert type(curve_array_list) == list and len(curve_array_list) > 0 and all(isinstance(curve, Curve) for curve in curve_array_list)
    time_grid = curve_array_list[0].T     # union of all the temporal grids
    for i in range(1, len(curve_array_list)):
        time_grid = np.union1d(time_grid, curve_array_list[i].T)
    return time_grid


def Max_forward_maturity(forward_curves_list: list, maturity: float) -> int:
    """
    Returns the index of the forward curve in the list that has the maximum forward price maturity date (in yrf).
    :param forward_curves_list (list[EquityForwardCurve]): list of forward curves.
    :param maturity (float): maturity date (in yrf).
    :return index (int): index of the forward curve in the list that has the maximum forward price maturity date (in yrf).
    """
    Ndim = len(forward_curves_list)
    F = np.zeros(Ndim)
    for i in range(Ndim):
        F[i] = forward_curves_list[i](maturity)
    return np.argmax(F)


def Min_mu_each_time(mu: Drift) -> np.ndarray:
    """
    Returns the indexes of the asset with minimum drift for each time step.
    :param mu (Drift): drift of the assets.
    :return index_min_mu (np.ndarray[int]): indexes of the asset with minimum drift for each time step.
    """
    n_times = len(mu.T)
    index_min_mu = np.zeros(n_times)
    for i in range(n_times):
        if i == 0:
            index_min_mu[i] = np.argmin(mu(0.))
        else:
            index_min_mu[i] = np.argmin(mu(mu.T[i-1]))
    return index_min_mu


def Min_mu_nu_each_time(mu: Drift, nu: CholeskyTDependent) -> tuple:
    """
    Returns the indexes of the asset with minimum drift/||nu|| for each time step. Moreover, it returns the
    allocation time_grid.
    :param mu (Drift): drift of the assets.
    :param nu (CholeskyTDependent): Cholesky decomposition of the covariance matrix of the assets.
    :return (tuple): indexes of the asset with minimum drift/||nu|| for each time step, and the allocation time_grid.
    """
    time_grid = mu.T
    time_grid = np.union1d(time_grid, nu.T)
    n_times = len(time_grid)
    index_min_mu_nu = np.zeros(n_times)
    for i in range(n_times):
        if i == 0:
            index_min_mu_nu[i] = np.argmin(mu(0.) / np.linalg.norm(nu(0.)))
        else:
            index_min_mu_nu[i] = np.argmin(mu(time_grid[i-1]) / np.linalg.norm(nu(time_grid[i-1])))
    return index_min_mu_nu, time_grid


def Markowitz_solution(mu: np.ndarray, nu: np.ndarray, sign: float) -> np.ndarray:
    """
    Closed form of the optimal strategy with free bounds.
    :param mu (np.ndarray[float]): drift of the assets.
    :param nu (np.ndarray[float]): Cholesky decomposition of the covariance matrix of the assets.
    :return(np.ndarray[float]): optimal strategy. The shape of the array is (n_assets,).
    """
    assert len(mu) == len(nu) == len(nu.T)
    covariance_matrix = nu@nu.T  # covariance matrix
    inverse_covariance_matrix = np.linalg.inv(covariance_matrix)  # inverse of covariance matrix
    norm = sign * 0.5 * np.linalg.norm((inverse_covariance_matrix @ mu) @ nu)
    return 0.5 * (1 / norm) * (inverse_covariance_matrix @ mu)


def loss_function(x: np.array, mu_value: np.ndarray, nu_value: np.ndarray) -> float:
    """
    Local drift to minimize.
    :param x (np.ndarray[float]): optimal strategy. The shape of the array is (n_assets,).
    :param mu_value (np.ndarray[float]): drift of the assets.
    :param nu_value (np.ndarray[float]): Cholesky decomposition of the covariance matrix of the assets.
    """
    return (x @ mu_value) / np.linalg.norm(x @ nu_value)


def optimization_only_long(mu_value: np.ndarray, nu_value: np.ndarray, n_trial: int = 1,
                           seed: int = 1, guess: np.ndarray = None) -> np.ndarray:
    """
    Constrained optimization with only long position and sum of weights equal to 1
    :param mu_value (np.ndarray[float]): drift of the assets.
    :param nu_value (np.ndarray[float]): Cholesky decomposition of the covariance matrix of the assets.
    :param n_trial (int): number of trials with different initial guess.
    :param seed (int): seed for the random number generator.
    :param guess (np.ndarray[float]): initial guess for the optimization. If it is provided then n_trial is ignored.
    the guess have the following shape (n_trial, n_assets).
    :return (np.ndarray[float]): optimal strategy. The shape of the array is (n_assets,).
    """
    assert len(mu_value) == len(nu_value) == len(nu_value.T)

    f = loss_function
    A = np.ones(len(mu_value))
    x_low = np.array([1.])
    x_up = np.array([1.])
    bounds = Bounds(np.zeros(len(mu_value)), np.ones(len(mu_value)))
    constraints = LinearConstraint(A, x_low, x_up)
    if guess is None:
        generator = np.random; generator.seed(seed)
        r = np.zeros((n_trial, len(mu_value)))
        valutation = np.zeros(n_trial)
        for i in range(n_trial):
            x0 = generator.uniform(0., 1., len(mu_value))  #initial position for the optimization algorithm
            res = minimize(f, x0, args=(mu_value, nu_value), constraints=constraints, bounds=bounds, method="SLSQP")
            r[i] = res.x
            valutation[i] = f(res.x, mu_value, nu_value)
        return r[np.argmin(valutation)]
    elif guess.ndim == 1:
        return minimize(f, guess, args=(mu_value, nu_value), constraints=constraints, bounds=bounds, method="SLSQP").x
    elif guess.ndim > 1:
        n_trial = len(guess)
        r = np.zeros((n_trial, len(mu_value)))
        valutation = np.zeros(n_trial)
        for i in range(n_trial):
            x0 = guess[i]
            res = minimize(f, x0, args=(mu_value, nu_value), constraints=constraints, bounds=bounds, method="SLSQP")
            r[i] = res.x
            valutation[i] = f(res.x, mu_value, nu_value)
        return r[np.argmin(valutation)]


def optimization_limit_position(mu_value: np.ndarray, nu_value: np.ndarray, limit_position: float,
                                n_trial: int, seed: int) -> np.ndarray:
    """
    Constrained optimization with the constraint |alpha_i|<limit_position for all i.
    :param mu_value (np.ndarray[float]): drift of the assets.
    :param nu_value (np.ndarray[float]): Cholesky decomposition of the covariance matrix of the assets.
    :param limit_position (float): limit position for each allocation strategy.
    :param n_trial (int): number of trials with different initial guess.
    :param seed (int): seed for the random number generator.
    :return (np.ndarray[float]): optimal strategy. The shape of the array is (n_assets,).
    """
    assert len(mu_value) == len(nu_value) == len(nu_value.T)
    generator = np.random; generator.seed(seed)
    f = loss_function
    cons = ({'type': 'ineq', 'fun': lambda x: -abs(x) + limit_position})
    r = np.zeros((n_trial, len(mu_value)))
    valutation = np.zeros(n_trial)
    for i in range(n_trial):
        x0 = generator.uniform(-limit_position, limit_position, len(mu_value))
        res = minimize(f, x0, args=(mu_value, nu_value), constraints=cons)
        r[i] = res.x
        valutation[i] = f(res.x, mu_value, nu_value)
    return r[np.argmin(valutation)]


def optimization_long_short_position(mu_value: np.ndarray, nu_value: np.ndarray, long_limit: float, short_limit: float,
                                     n_trial: int, seed: int) -> np.ndarray:
    """
    Constrained optimization where the sum of the long (alpha_i>0) allocation positions must be lower than long_limit,
    while the sum of the short (alpha_i<0) allocation positions must be lower than short_limit.
    :param mu_value (np.ndarray[float]): drift of the assets.
    :param nu_value (np.ndarray[float]): Cholesky decomposition of the covariance matrix of the assets.
    :param long_limit (float): limit position for the sum of the long allocation strategy.
    :param short_limit (float): limit position for the sum of the short allocation strategy.
    :param n_trial (int): number of trials with different initial guess.
    :param seed (int): seed for the random number generator.
    :return (np.ndarray[float]): optimal strategy. The shape of the array is (n_assets,).
    """
    assert len(mu_value) == len(nu_value) == len(nu_value.T)
    generator = np.random; generator.seed(seed)
    f = loss_function
    constraints = ({'type': 'ineq', 'fun': lambda x: -np.sum(x[np.where(x > 0)[0]]) + long_limit},
                   {'type': 'ineq', 'fun': lambda x: -abs(np.sum(x[np.where(x < 0)[0]])) + short_limit})
    r = np.zeros((n_trial, len(mu_value)))
    valutation = np.zeros(n_trial)
    for i in range(n_trial):
        x0 = generator.uniform(-short_limit, long_limit, len(mu_value))
        res = minimize(f, x0, args=(mu_value, nu_value), constraints=constraints)
        r[i] = res.x
        valutation[i] = f(res.x, mu_value, nu_value)
    return r[np.argmin(valutation)]
