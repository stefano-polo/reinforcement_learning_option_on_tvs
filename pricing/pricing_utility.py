import numpy as np


def quad_piecewise(piece_wise_function, time_grid_of_piece_wise_function: np.ndarray, lower_int: float,
                   upper_int: float or np.ndarray, vectorized: bool = False) -> float or np.ndarray:
    """
    Function that integrates a piecewise constant function (right open).
    :param piece_wise_function: function to be integrated
    :param time_grid_of_piece_wise_function: time grid where the piecewise function is defined.
    :param lower_int (float): lower integration bound.
    :param upper_int (float): upper integration bound.
    :param vectorized (bool, default=False): boolean that indicates if the function is vectorial (True) or scalar (False).
    :return: the integral of the piecewise function.
    """
    t_in = float(lower_int)
    t_fin = float(upper_int)
    time_grid = np.float64(time_grid_of_piece_wise_function)
    if t_in == t_fin:
        return 0
    if t_fin in time_grid:
        time_grid = time_grid[np.where(time_grid <= t_fin)[0]]
    if t_in in time_grid:
        time_grid = time_grid[np.where(time_grid >= t_in)[0]]
    if t_in not in time_grid:
        time_grid = time_grid[np.where(time_grid > t_in)[0]]
        time_grid = np.insert(time_grid, 0, t_in)
    if t_fin not in time_grid:
        time_grid = time_grid[np.where(time_grid < t_fin)[0]]
        time_grid = np.insert(time_grid, len(time_grid), t_fin)
    if vectorized:
        y = np.array([])
        for i in range(1, len(time_grid)):
            y = np.append(y, piece_wise_function(time_grid[i]))
    else:
        y = piece_wise_function(time_grid[:-1])

    dt = np.diff(time_grid)
    return np.sum(y * dt)


def get_euler_grid(fixings: np.ndarray, n_intervals: int) -> tuple:
    """
    Function that takes a vector of fixings (expressed in yrf) and returns a tuple of two vectors:
    - the first vector contains the Euler grid (expressed in yrf) of the fixings (among each fixings
    the Euler grid is the vector of the fixings divided by n_intervals)
    tuple of the grid of the Euler method and the number of intervals
    :param fixings (np.ndarray[float]): vector of fixings (expressed in yrf)
    :param n_intervals (int): number of intervals between the fixings.
    :return (tuple): tuple of the grid of the Euler method (np.ndarray with shape: len(fixings)*n_intervals)
    and time intervals of the grids between the fixings (np.ndarray with shape: len(fixings)-1).
    """
    time_grid = np.array([])
    dt = 0.0
    for i in range(len(fixings)):
        if i == 0:
            dt = np.array([fixings[i] / n_intervals])
            time_grid = np.append(time_grid, np.linspace(dt[i], fixings[i], n_intervals))
        else:
            dt = np.append(dt, (fixings[i] - fixings[i - 1]) / n_intervals)
            time_grid = np.append(time_grid, np.linspace(fixings[i - 1] + dt[i], fixings[i], n_intervals))
    return time_grid, dt


def DayCountConversion(reference_date: int, date: np.ndarray or int, convention: str) -> np.ndarray or int:
    """
    Function that takes a reference_date (expressed in days int) and an array of dates (expressed in days int)
    and returns them expressed in year fractions (float) according to the convention (available ACT_365, ACT_360, and None).
    If the convention is not provided then the function treats the dates and the reference_date as if they are expressed in yrf.
    :param reference_date (int): reference date expressed in days
    :param date (np.ndarray[int] or int): array of dates expressed in days
    :param convention (str): convention to be used (ACT_365, ACT_360, None)
    :return: date expressed in year fractions (float or np.ndarray[float]) according to the convention
    """
    if convention == 'ACT_365':
        return ACT_365(reference_date, date)
    elif convention == 'ACT_360':
        return ACT_360(reference_date, date)
    elif convention is None:
        return np.fabs(date - reference_date)
    else:
        raise ValueError('The selected convention is not available (available ACT_365, ACT_360, None)')


def ACT_365(reference_date: int, date: np.ndarray or int) -> np.ndarray or int:
    """
    Function that takes a reference_date (expressed in days int) and one or an array of dates (expressed in days int)
    and returns them expressed in year fractions (float) according to the ACT/365 convention.
    :param reference_date (int): reference date expressed in days
    :param date (np.ndarray[int] or int): array of dates expressed in days
    :return: date expressed in year fractions (float or np.ndarray[float]) according to the ACT/365 convention
    """
    return np.abs(date - reference_date) / 365.0


def ACT_360(reference_date: np.array or float, date: np.array or float) -> np.array or float:
    """
    Function that takes a reference_date (expressed in days int) and one or an array of dates (expressed in days int)
    and returns them expressed in year fractions (float) according to the ACT/360 convention.
    :param reference_date (int): reference date expressed in days
    :param date (np.ndarray[int] or int): array of dates expressed in days
    :return: date expressed in year fractions (float or np.ndarray[float]) according to the ACT/360 convention
    """
    return np.abs(date - reference_date) / 360.0

