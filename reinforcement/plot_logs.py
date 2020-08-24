from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt


def plot_rolling(x, y, window, title='reward', std_mul=2.33):
    """Plot moving average with error bands."""

    y_roll = y.rolling(window)
    y_mean = y_roll.mean()
    y_std = y_roll.std() / sqrt(window)
    y_lo = y_mean - std_mul * y_std
    y_hi = y_mean + std_mul * y_std
    plt.plot(x, y_mean, label=title)
    plt.fill_between(x, y_lo, y_hi, alpha=0.1)
    print("max y = {} at x = {}".format(y_mean.max(), x[np.argmax(y_mean)]))
