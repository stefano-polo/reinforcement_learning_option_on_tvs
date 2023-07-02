import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from plot_utilities import join_curves, plot_rolling, pu

######################################################## SCRIPT INPUT ##############################################################################################

discount_factor = (
    1.0097815717231273  # discount factor used to discount the learning curves
)
n_sigma = 2.55  # number of standard deviations to plot
join_learning = True  # if True then the learning curves provided in WHICH_LOGS are joined together. If False then each learning curve is plotted separately.
labels_fontsize = 12  # font size for the labels of the learning curves plots
X_AXIS_TIMESTEPS = False  # if True then the x-axis is in time-steps, if False then the x-axis is in episodes
MOVING_AVERAGE_WINDOW = int(
    1000
)  # window of the moving average used to smooth the learning curves (measured in episodes)
LOG_FOLDER = "./logs/TVS_LV-v2/"  # root folder of the environment log files
WHICH_LOGS = [  # ("name of the output folder", "legend label"). If join_learning is True then the legend label is ignored.
    (
        "ppo2_1e6_5x8_3e-4_trainingseed45891_beta0.7_copy_tanhpricing_prova",
        "learning_curve_1",
    ),
    (
        "ppo2_1e5_5x8_3e-4_trainingseed45891_beta0.7_copy_tanhpricing_prova",
        "learning_curve_2",
    ),
]

######################################################################################################################################################################

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
if join_learning:
    steps_joined, rewards_joined = join_curves(LOG_FOLDER, WHICH_LOGS, X_AXIS_TIMESTEPS)
    plot_rolling(
        steps_joined,
        rewards_joined * discount_factor,
        MOVING_AVERAGE_WINDOW,
        WHICH_LOGS[0][1],
        n_sigma,
    )
else:
    for log, title in WHICH_LOGS:
        results = pu.load_results(LOG_FOLDER + log)
        r = results[0]
        steps = (
            np.cumsum(r.monitor.l) if X_AXIS_TIMESTEPS else np.array(r.monitor.l.index)
        )
        rewards = r.monitor.r  # episode reward
        plot_rolling(
            steps, rewards * discount_factor, MOVING_AVERAGE_WINDOW, title, n_sigma
        )


plt.axhline(
    (0.04149584726740925 + n_sigma * 5.089556147059726e-05),
    color="red",
    lw=2.5,
    linestyle="-.",
    label="Baseline Strategy",
)
plt.axhline(
    (0.04149584726740925 - n_sigma * 5.089556147059726e-05),
    color="red",
    lw=2.5,
    linestyle="-.",
)
plt.legend(prop={"size": labels_fontsize})
plt.grid(True)
plt.xlabel("Time step" if X_AXIS_TIMESTEPS else "Episode", fontsize=labels_fontsize)
plt.ylabel("Episode reward rolling avg [EUR]", fontsize=labels_fontsize)
plt.title("Learning curve", fontsize=labels_fontsize)
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.xaxis.set_major_formatter(formatter)
plt.show()
