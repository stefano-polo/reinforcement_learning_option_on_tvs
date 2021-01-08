import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sqrt, log
import tensorflow as tf
from plot_agent import plot, build_args,model_creation, plot3d
strategylong = True

Seed = 19

n_equity = 2
time_dep = True
constraint = "only_long"  #only_long, free, long_short_limit
model = 'LV'
sum_short=50./100.
sum_long=50./100.
input_normalize = 2
market_data=1
# configurations: which environment and agents
PLOT_VALUE = False  # otherwise plots actions
VARIABLE_INDEXES = (2,)  # which dimension to vary
VARIABLE_POINTS = 12
ENV = 'TVS_LV-v0'
REFERENCE_STATE = np.array([np.log(1), np.log(1),1/12])
X_MAX = 1
ACTION_SPACE_DESCRIPTION = '[log(S/S0), t]'
AGENTS = [
    #(build_args(ENV, 'ppo2', '3e7', '2', '3', '3e-4', custom_suffix='long_change_2long_two_asset_change_beta0.7'), {'action_grid_size': 0}, 'PPO')
(build_args(ENV, 'ppo2', '1e8', '1', '6', '3e-4', value_network='copy',beta='0.7',custom_suffix='long_month_observation_seed1364798666_'), {'action_grid_size': 0}, 'PPO')

]

reference_str = list(map(str, REFERENCE_STATE))
for ivar in VARIABLE_INDEXES:
    reference_str[ivar] = ':'
z_name = 'Value function' if PLOT_VALUE else 'Actions'
title = '{} at {} = ['.format(z_name, ACTION_SPACE_DESCRIPTION) + ', '.join(reference_str) + ']'

fig, ax = plt.subplots(1, 1, figsize=(8, 5))


for (arg, env_args, lbl) in AGENTS:
    g = tf.Graph()
    sess = tf.InteractiveSession(graph=g)
    with g.as_default():
        plot(arg, PLOT_VALUE, env_args, REFERENCE_STATE, VARIABLE_INDEXES, VARIABLE_POINTS, 
        title, lbl, X_MAX, all_time_dep = time_dep, seed=Seed, strategy_constraint= constraint,
        how_long=sum_long, how_short=sum_short, N_equity = n_equity, normalized=input_normalize, 
        market=market_data, pricing_model=model)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
plt.savefig("Action_RL_1x4_seed"+str(Seed)+".png",bbox_inches='tight',dpi=900)
plt.show()
