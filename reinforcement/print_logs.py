from plot_logs import plot_rolling, pu, join_curves
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

font = 13
LOG_FOLDER = './logs/TVS_simple-v0/'   #in which folder I find the file to plot
WHICH_LOGS = [
    ('ppo2_1e8_4x6_3e-4long_month_observation_beta0.7_2.5variance_seed1353720613', 'seed=1353720613'),
]
X_AXIS_TIMESTEPS = True  # otherwise: episodes
WINDOW = int(5e5)  # measured in episodes  10000
join_learning = False   
fig, ax = plt.subplots(1, 1, figsize=(8, 5))


if join_learning:
    steps_joined, rewards_joined = join_curves(LOG_FOLDER, WHICH_LOGS, X_AXIS_TIMESTEPS)
    plot_rolling(steps_joined, rewards_joined, WINDOW, r"$\beta$=0.7 4x6")
else:
    for log, title in WHICH_LOGS:
        results = pu.load_results(LOG_FOLDER + log)
        r = results[0]
        steps = np.cumsum(r.monitor.l) if X_AXIS_TIMESTEPS else np.array(r.monitor.l.index)
        rewards = r.monitor.r           # episode reward
        plot_rolling(steps, rewards, WINDOW, title)

formatter = ticker.ScalarFormatter(useMathText=True) #scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.axhline(0.026816785924320553, color='red', linestyle = '--',label="3 asset option price")
plt.axhline(0.02590927186763423,color='blue',linestyle = '--',label="2 asset option price")
plt.legend()
plt.grid(True)
plt.xlabel('Time step' if X_AXIS_TIMESTEPS else 'Episode',fontsize=font)
plt.ylabel('Episode reward rolling avg [EUR]',fontsize=font)
plt.title('Learning curve',fontsize=font)
plt.savefig("Learning_curve_market_original_4x4_8x5.png",bbox_inches='tight',dpi=900)
plt.show()
