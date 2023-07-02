import numpy as np


def n_sphere_to_cartesian(radius: float, angles: np.ndarray) -> np.ndarray:
    """
    Function that takes n-spherical coordinates and convert into n-cartesian coordinates
    angles: the n-2 values between [0,\pi) and last one between [0,2\pi)
    :param radius (float): the radius of the n-sphere
    :param angles (np.ndarray): the angles of the n-sphere
    :return: the n-cartesian coordinates (shape: (len(angles),))
    """
    a = np.concatenate((np.array([2 * np.pi]), angles))
    si = np.sin(a)
    si[0] = 1
    si = np.cumprod(si)
    co = np.cos(a)
    co = np.roll(co, -1)
    return si * co * radius


def sign_renormalization(
    vec: np.ndarray, sum_plus: float, sum_minus: float
) -> np.ndarray:
    """
    Renormalize positive and negative elements of an array separately if necessary such
    that sum of positive is equal to sum_plus and the abs sum of negative is equal to sum_neg
    :param vec: the array to be renormalized
    :param sum_plus: the sum of positive elements
    :param sum_minus: the sum of negative elements
    :return: the renormalized array (shape: (len(vec),))
    """
    pos = vec >= 0
    neg = np.invert(pos)
    p_s = np.sum(vec[pos])
    n_s = abs(np.sum(vec[neg]))
    if p_s > sum_plus:
        vec[pos] = (vec[pos] / p_s) * sum_plus
    if n_s > sum_minus:
        vec[neg] = (vec[neg] / n_s) * sum_minus
    return vec


def build_allocation_time_grid(
    maturity: float,
    allocation_frequency: str = "monthly",
    day_count_convention: str = "ACT_365",
):
    """
    Builds a time grid for the allocation of a portfolio
    :param maturity: the maturity of the portfolio in years (it must be an integer number of years).
    :param allocation_frequency: the frequency of the allocation
    :param day_count_convention: the day count convention
    :return: a numpy array of the time grid
    """
    if day_count_convention == "ACT_365":
        number_days_in_year = 365.0
    elif day_count_convention == "ACT_360":
        number_days_in_year = 360.0
    else:
        raise ValueError("Unknown day count convention (available: ACT_365, ACT_360)")

    if maturity % 1.0 != 0:
        raise ValueError("Maturity must be an integer number of years")

    if allocation_frequency == "monthly":
        month_days_lenght = np.array(
            [31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0]
        )
        months = month_days_lenght
        if maturity > 1.0:
            for i in range(int(maturity) - 1):
                months = np.append(months, month_days_lenght)
        observation_time_grid = np.cumsum(months) / number_days_in_year
        n_euler_grid_points = 60
        state_index_grid = np.arange(int(12 * maturity) + 1) * n_euler_grid_points
    elif allocation_frequency == "daily":
        observation_time_grid = np.linspace(
            1.0 / number_days_in_year, maturity, int(maturity * number_days_in_year)
        )
        n_euler_grid_points = 2
        state_index_grid = np.arange(int(365 * maturity) + 1) * n_euler_grid_points

    observation_time_grid = np.append(0.0, observation_time_grid)
    return observation_time_grid, state_index_grid, n_euler_grid_points
