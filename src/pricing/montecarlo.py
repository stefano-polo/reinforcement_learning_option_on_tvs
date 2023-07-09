import numpy as np


def MC_results(mc_paths: np.ndarray) -> tuple:
    """
    Returns mean and uncertainty of a Monte Carlo simulation.
    :param mc_paths (np.ndarray[float]): Monte Carlo simulation. It can be a vector or a matrix.
    The mean and uncertainty are returned for each column.
    :return: mean, uncertainty.
    """
    mean = np.mean(mc_paths, axis=0)
    std = np.std(mc_paths, axis=0) / np.sqrt(len(mc_paths))
    return mean, std


def MC_Data_Blocking(mc_paths: np.ndarray or np.array, n_blocks: int) -> tuple:
    """
    Data blocking algorithm to check the convergence of the Monte Carlo simulation.
    :param mc_paths (np.ndarray[float]): Monte Carlo simulation paths. It can be a vector or a matrix (rows are the paths).
    :param n_blocks (int): Number of blocks.
    :return (tuple): indexes of the blocks (the first index stats for n_blocks, while second n_blocks*2 and so on),
    mean of the single block, error of the given block.
    """
    if mc_paths.ndim == 1:
        f = MC_Analisys_vector
    elif mc_paths.ndim == 2:
        f = MC_Analisys_matrix
    else:
        raise ValueError("The input must be a vector or a matrix")
    return f(mc_paths, n_blocks)


def MC_Analisys_matrix(
    mc_paths: np.ndarray, n_blocks: int
) -> tuple:  # method to check convergence of the MC result
    """
    Data blocking method for a matrix
    :param mc_paths (np.ndarray[float]): Monte Carlo simulation paths in matrix format. The mean is performed on each column.
    :param n_blocks (int): Number of blocks.
    :return (tuple): indexes of the blocks (the first index stats for n_blocks, while second n_blocks*2 and so on),
    mean of the single block, error of the given block.
    """
    assert mc_paths.ndim == 2, "The input must be a matrix"
    n_paths = len(mc_paths)
    n_dims = len(mc_paths.T)
    n_path_per_blocks = int(
        n_paths / n_blocks
    )  # Number of throws in each block, please use for n_paths a multiple of n_blocks
    ave = np.zeros((n_blocks, n_dims))
    av2 = np.zeros((n_blocks, n_dims))
    sum_prog = np.zeros((n_blocks, n_dims))
    sum2_prog = np.zeros((n_blocks, n_dims))
    err_prog = np.zeros((n_blocks, n_dims))
    x = np.arange(n_blocks)
    x *= n_path_per_blocks
    for m in range(n_dims):
        for i in range(n_blocks):
            s = 0.0
            for j in range(n_path_per_blocks):
                k = j + i * n_path_per_blocks
                s += mc_paths[k, m]
            ave[i, m] = s / n_path_per_blocks
            av2[i, m] = (ave[i, m]) ** 2

    for m in range(n_dims):
        for i in range(n_blocks):
            for j in range(i + 1):
                sum_prog[i, m] += ave[j, m]
                sum2_prog[i, m] += av2[j, m]
            sum_prog[i, m] /= i + 1  # Cumulative average
            sum2_prog[i, m] /= i + 1  # Cumulative square average
            err_prog[i, m] = error(sum_prog, sum2_prog, i, m)  # Statistical uncertainty
    return x, sum_prog, err_prog


def MC_Analisys_vector(
    mc_paths: np.ndarray, n_blocks: int
) -> tuple:  # method to check convergence of the MC result
    """
    Data blocking method for a vector.
    :param mc_paths (np.ndarray[float]): Monte Carlo simulation paths in vector format.
    :param n_blocks (int): Number of blocks.
    :return (tuple): indexes of the blocks (the first index stats for n_blocks, while second n_blocks*2 and so on),
    mean of the single block, error of the given block.
    """
    assert mc_paths.ndim == 1, "The input must be a vector"
    n_paths = len(mc_paths)
    n_path_per_blocks = int(
        n_paths / n_blocks
    )  # Number of throws in each block, please use for n_paths a multiple of n_blocks
    ave = np.zeros(n_blocks)
    av2 = np.zeros(n_blocks)
    sum_prog = np.zeros(n_blocks)
    sum2_prog = np.zeros(n_blocks)
    err_prog = np.zeros(n_blocks)
    x = np.arange(n_blocks)
    x *= n_path_per_blocks
    for i in range(n_blocks):
        s = 0
        for j in range(n_path_per_blocks):
            k = j + i * n_path_per_blocks
            s += mc_paths[k]
        ave[i] = s / n_path_per_blocks
        av2[i] = (ave[i]) ** 2

    for i in range(n_blocks):
        for j in range(i + 1):
            sum_prog[i] += ave[j]
            sum2_prog[i] += av2[j]
        sum_prog[i] /= i + 1  # Cumulative average
        sum2_prog[i] /= i + 1  # Cumulative square average
        err_prog[i] = error_vector(sum_prog, sum2_prog, i)  # Statistical uncertainty
    return x, sum_prog, err_prog


def error(
    AV: np.ndarray, AV2: np.ndarray, n: int, m: int
):  # Function for statistical uncertainty estimation
    """
    Error Function for matrix
    """
    if n == 0:
        return 0
    else:
        return np.sqrt((AV2[n, m] - AV[n, m] ** 2) / n)


def error_vector(
    AV: np.ndarray, AV2: np.ndarray, n: int
):  # Function for statistical uncertainty estimation
    """
    Error Function for vector
    """
    if n == 0:
        return 0
    else:
        return np.sqrt((AV2[n] - AV[n] ** 2) / n)
