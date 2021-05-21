from plot_logs import plot_rolling, pu, join_curves
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

font = 13
n_sigma = 1.5
strategy = 'free'

LOG_FOLDER = './logs/TVS_LV_newreward-v0/'   #in which folder I find the file to plot
WHICH_LOGS = [  #("name of the output folder", "legend label")
('ppo2_9e7_2x8_3e-4_trainingseed20319_beta0.7_copyfreefrombaseline_displacedmarket_2assets_monthgrid_maturity2','RL'),
]
X_AXIS_TIMESTEPS = 0  # otherwise: episodes
WINDOW = int(3e5)  # measured in episodes  10000
join_learning = 0  
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

discount = 1.0097815717231273

if join_learning:
    steps_joined, rewards_joined = join_curves(LOG_FOLDER, WHICH_LOGS, X_AXIS_TIMESTEPS)
    plot_rolling(steps_joined, rewards_joined*discount, WINDOW, r"RL")  #discount factor
else:
    for log, title in WHICH_LOGS:
        results = pu.load_results(LOG_FOLDER + log)
        r = results[0]
        steps = np.cumsum(r.monitor.l) if X_AXIS_TIMESTEPS else np.array(r.monitor.l.index)
        rewards = r.monitor.r           # episode reward
        plot_rolling(steps, rewards, WINDOW, title)


if strategy=="only_long":
    plt.axhline( 0.014129190929724296+2.5447894383989902e-05*n_sigma, color='red', lw=2.5,linestyle = '--',label="BS Strategy")
    plt.axhline( 0.014129190929724296-2.5447894383989902e-05*n_sigma, color='red',lw=2.5, linestyle = '--')
    plt.axhline(0.014784083837497124+3.2454334287712235e-05*n_sigma, color='green',lw=2.5, linestyle = '-.',label="Baseline Strategy")
    plt.axhline(0.014784083837497124-3.2454334287712235e-05*n_sigma, color='green',lw=2.5, linestyle = '-.')
        
elif strategy=='free':
    plt.axhline(0.040947694174563704 +n_sigma*4.325646589751639e-05, color='red', lw=2.5,linestyle = '--',label="BS Strategy")
    plt.axhline( 0.040947694174563704-n_sigma*4.325646589751639e-05, color='red',lw=2.5, linestyle = '--')
    plt.axhline((0.04108567123304307+n_sigma*5.03916718605177e-05)*discount, color='green',lw=2.5, linestyle = '-.',label="Baseline Strategy")
    plt.axhline((0.04108567123304307-n_sigma* 5.03916718605177e-05)*discount, color='green',lw=2.5, linestyle = '-.')
plt.legend()
plt.grid(True)
plt.xlabel('Time step' if X_AXIS_TIMESTEPS else 'Episode',fontsize=font)
plt.ylabel('Episode reward rolling avg [EUR]',fontsize=font)
plt.title('Learning curve',fontsize=font)
formatter = ticker.ScalarFormatter(useMathText=True) #scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
ax.xaxis.set_major_formatter(formatter)
#ax.yaxis.set_major_formatter(formatter)
#plt.savefig("Learning_curve_market_original_4x4_8x5.png",bbox_inches='tight',dpi=900)
plt.show()
