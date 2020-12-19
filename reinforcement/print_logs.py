from plot_logs import plot_rolling, pu, join_curves
import matplotlib.pyplot as plt
import numpy as np
LOG_FOLDER = './logs/TVS_simple-v0/'   #in which folder I find the file to plot
WHICH_LOGS = [
    ('ppo2_1e8_4x6_3e-4long_month_observation_beta0.7_2.5variance_seed1353720613', 'seed=1353720613'),
]
X_AXIS_TIMESTEPS = True  # otherwise: episodes
WINDOW = int(5e5)  # measured in episodes  10000
join_learning = False   
plt.figure(figsize=(8,5))

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

plt.axhline(0.026816785924320553, color='red', linestyle = '--',label="3 asset option price")
plt.axhline(0.02590927186763423,color='blue',linestyle = '--',label="2 asset option price")
plt.legend()
plt.xlabel('time step' if X_AXIS_TIMESTEPS else 'episode',fontsize=13)
plt.ylabel('episode reward rolling avg [EUR]',fontsize=13)
plt.title('Learning curve',fontsize=13)
plt.savefig("Learning_curve_market_original_4x4_8x5.png",bbox_inches='tight',dpi=900)
plt.show()
