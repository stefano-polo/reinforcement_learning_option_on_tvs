from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import pandas as pd

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



def join_curves(LOG_FOLDER, WHICH_LOGS, X_AXIS_TIMESTEPS):
    i = 0
    for log, title in WHICH_LOGS:
        results = pu.load_results(LOG_FOLDER + log)
        r = results[0]
        steps = np.cumsum(r.monitor.l) if X_AXIS_TIMESTEPS else np.array(r.monitor.l.index)
        rewards = r.monitor.r
        if i == 0:
            if X_AXIS_TIMESTEPS:
                steps_joined = steps.values
            else:
                steps_joined = steps
            rewards_joined = rewards
        else:
            rewards.index = rewards.index+len(rewards_joined)
            rewards_joined = pd.concat([rewards_joined,rewards])
            if X_AXIS_TIMESTEPS:
                steps_joined = np.concatenate((steps_joined,steps.values+steps_joined[-1]))
            else:
                steps_joined = np.concatenate((steps_joined,steps+steps_joined[-1]))
        i+=1
    return steps_joined,rewards_joined