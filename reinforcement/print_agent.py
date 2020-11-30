import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sqrt, log
from plot_logs import plot_rolling, pu, join_curves
import tensorflow as tf
from plot_agent import plot, build_args,model_creation, plot3d
strategylong = True

Seed = 81

n_equity = 3
time_dep = True
constraint = "long_short_limit"  #only_long, free, long_short_limit
sum_short=50./100.
sum_long=50./100.
input_normalize = 2
market_data=False
# configurations: which environment and agents
PLOT_VALUE = False  # otherwise plots actions
VARIABLE_INDEXES = (3,)  # which dimension to vary
VARIABLE_POINTS = 12
ENV = 'TVS_simple-v0'
REFERENCE_STATE = np.array([np.log(1),np.log(1), np.log(1),1/12])
X_MAX = 1
ACTION_SPACE_DESCRIPTION = '[log(S/S0), t]'
AGENTS = [
    #(build_args(ENV, 'ppo2', '3e7', '2', '3', '3e-4', custom_suffix='long_change_2long_two_asset_change_beta0.7'), {'action_grid_size': 0}, 'PPO')
(build_args(ENV, 'ppo2', '1e8', '4', '6', '3e-4', value_network='copy',beta='0.7',custom_suffix='long_month_observation_beta0.7_2.5variance_seed517043487_restarted'), {'action_grid_size': 0}, 'PPO')

]

reference_str = list(map(str, REFERENCE_STATE))
for ivar in VARIABLE_INDEXES:
    reference_str[ivar] = ':'
z_name = 'Value function' if PLOT_VALUE else 'Actions'
title = '{} at {} = ['.format(z_name, ACTION_SPACE_DESCRIPTION) + ', '.join(reference_str) + ']'

plt.figure(figsize=(8,5))

for (arg, env_args, lbl) in AGENTS:
    g = tf.Graph()
    sess = tf.InteractiveSession(graph=g)
    with g.as_default():
        plot(arg, PLOT_VALUE, env_args, REFERENCE_STATE, VARIABLE_INDEXES, VARIABLE_POINTS, title, lbl, X_MAX, all_time_dep = time_dep, seed=Seed, strategy_constraint= constraint,how_long=sum_long, how_short=sum_short, N_equity = n_equity, normalized=input_normalize, market=market_data)
#plt.savefig("Action_RL_2x6_8x5_seed"+str(Seed)+".png",bbox_inches='tight',dpi=900)
plt.show()
