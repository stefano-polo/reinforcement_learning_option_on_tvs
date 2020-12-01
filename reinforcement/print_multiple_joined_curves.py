from plot_logs import plot_rolling, pu, join_curves
import matplotlib.pyplot as plt
import numpy as np

LOG_FOLDER = './logs/TVS_simple-v0/'   #in which folder I find the file to plot

X_AXIS_TIMESTEPS = False  # otherwise: episodes
WINDOW = int(8e5)  # measured in episodes  10000

WHICH_LOGS = [
  ('ppo2_3e8_1x6_3e-4long_month_observation_beta0.7_2.5variance_seed1733884390','1x4 beta'),
    ('ppo2_2e8_1x6_3e-4long_month_observation_beta0.7_2.5variance_seed1733884390_restarted','1x4 beta'),
]

plt.figure(figsize=(8,5))
steps_joined, rewards_joined = join_curves(LOG_FOLDER, WHICH_LOGS, X_AXIS_TIMESTEPS)
plot_rolling(steps_joined, rewards_joined, WINDOW, r"1x6")


WHICH_LOGS = [
  ('ppo2_3e8_2x6_3e-4long_month_observation_beta0.7_2.5variance_seed1733884390','1x4 beta'),
    ('ppo2_1e8_2x6_3e-4long_month_observation_beta0.7_2.5variance_seed1733884390_restarted','1x4 beta'),
    ('ppo2_1e8_2x6_3e-4long_month_observation_beta0.7_2.5variance_seed1733884390_restarted2','1x4') 
]

steps_joined, rewards_joined = join_curves(LOG_FOLDER, WHICH_LOGS, X_AXIS_TIMESTEPS)
plot_rolling(steps_joined, rewards_joined, WINDOW, r"2x6")

WHICH_LOGS = [
  ('ppo2_2e8_3x6_3e-4long_month_observation_beta0.7_2.5variance_seed3769704067', '0.45'),
    ('ppo2_1e8_3x6_3e-4long_month_observation_beta0.7_2.5variance_seed3769704067_restarted','0.45'),
   
]

steps_joined, rewards_joined = join_curves(LOG_FOLDER, WHICH_LOGS, X_AXIS_TIMESTEPS)
plot_rolling(steps_joined, rewards_joined, WINDOW, r"3x6")

WHICH_LOGS = [
  ('ppo2_2e8_4x6_3e-4long_month_observation_beta0.7_2.5variance_seed3769704067', '0.7'),
    ('ppo2_1e8_4x6_3e-4long_month_observation_beta0.7_2.5variance_seed3769704067_restarted','0.7'),
     ('ppo2_1e8_4x6_3e-4long_month_observation_beta0.7_2.5variance_seed3769704067_restarted','0.7'),
   
]
steps_joined, rewards_joined = join_curves(LOG_FOLDER, WHICH_LOGS, X_AXIS_TIMESTEPS)
plot_rolling(steps_joined, rewards_joined, WINDOW, r"4x6")


plt.axhline(0.026816785924320553, color='red', linestyle = '--',label="Option price")
plt.legend()
plt.xlabel('time step' if X_AXIS_TIMESTEPS else 'episode',fontsize=13)
plt.ylabel('episode reward rolling avg [EUR]',fontsize=13)
plt.title('Learning curve',fontsize=13)
plt.savefig("Learning_curve_all_8x5.png",bbox_inches='tight',dpi=900)
plt.show()