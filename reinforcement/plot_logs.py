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


LOG_FOLDER = './logs/VanillaOption-v0/'   #in which folder I find the file to plot
WHICH_LOGS = [
    ('ppo2_1e6_5x4_3e-4my_code', 'PPO')      #which files I plot: it's a list of pair (name of file and legend) (list because maybe you would like to compare different plots from different algorithms)
    #('deepq_1e8_3x4_3e-4_grid100', 'DeepQ batch 32'),
    #('deepq_1e8_3x4_3e-4_batch2048_grid100_nocheck', 'DeepQ batch 2048'),
    #('deepq_1e8_3x4_3e-4_cpepisodes10000_tgtfreq100000_updfreq250_batch7500_grid100', 'DeepQ by episodes'),
]
X_AXIS_TIMESTEPS = False  # otherwise: episodes
WINDOW = 10000  # measured in episodes

for log, title in WHICH_LOGS:
    results = pu.load_results(LOG_FOLDER + log)
    r = results[0]
    steps = np.cumsum(r.monitor.l) if X_AXIS_TIMESTEPS else np.array(r.monitor.l.index)
    rewards = r.monitor.r           # episode reward
    plot_rolling(steps, rewards, WINDOW, title)

plt.legend()
plt.xlabel('time step' if X_AXIS_TIMESTEPS else 'episode')
plt.ylabel('episode reward rolling avg')
plt.title('Learning curves')
plt.show()
