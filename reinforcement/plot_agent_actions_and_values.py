from plot_utilities import plot_one_episode_actions, plot_action_space
import numpy as np


######################################################## SCRIPT INPUT ##############################################################################################

plot_one_episode_action = False # if True, plot one episode action else plot a section of the action space or the value function
env_id = "TVS_LV-v2"
rl_model_parameters_dict = {"rl_algo": "ppo2",
                 "training_timesteps": "3e4",
                 "num_layers": "5",
                 "num_hidden": "8",
                 "learning_rate": "3e-4",
                 "activation": "tanh",
                 "value_network": "copy",
                 "beta": "0.7",
                 "custom_suffix": "_trainingseed45891_beta0.7_copy_tanhpricing",
                 "seed": "14"}

# parameters for plotting the action space or the value function
if not plot_one_episode_action:
    plot_value_function = False  # if false then plot the action space of the agent
    reference_state = np.array([np.log(1), np.log(1), 1.0, 1./12.])  # reference state where display the action/value space
                                                                     # block = [log(S_1(t)/F(t)), log(S_2(t)/F(t)), I/I_0, t]
    variable_indexes_to_vary = [0, 1]  # which state variable to vary in the plot (available 0: log(S_1(t)/F(t)), 1: log(S_2(t)/F(t)), 2: I/I_0, 3: t)
                                       # variable_indexes_to_vary must be a tuple or list of length 2 (3d plot) or 1 (2d plot)
    number_of_points = 20  # number of points to plot in the action/value space

################################################################################################################################################################

if plot_one_episode_action:
    plot_one_episode_actions(env_id, rl_model_parameters_dict)
else:
    plot_action_space(env_id, rl_model_parameters_dict, plot_value_function, reference_state, variable_indexes_to_vary, number_of_points)