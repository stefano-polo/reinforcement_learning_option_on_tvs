from math import sqrt

import envs.fe_envs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from baselines.common import plot_util as pu
from baselines.common.cmd_util import common_arg_parser
from baselines.run import parse_cmdline_kwargs, remove_train_noise, train


def plot_rolling(
    x: pd.Series,
    y: pd.Series,
    window: int,
    title: str = "reward",
    std_mul: float = 2.33,
) -> None:
    """
    Plot moving average with error bands of a pandas Series.
    :param x (pd.Series): x-axis values.
    :param y (pd.Series): y-axis values.
    :param window (int): window size for rolling mean.
    :param title (str): plot title.
    :param std_mul (float): multiplier for standard deviation error bars.
    """
    y_roll = y.rolling(window)
    y_mean = y_roll.mean()
    y_std = y_roll.std() / sqrt(window)
    y_lo = y_mean - std_mul * y_std
    y_hi = y_mean + std_mul * y_std
    plt.plot(x, y_mean, label=title)
    plt.fill_between(x, y_lo, y_hi, alpha=0.1)
    print("max y = {} at x = {}".format(y_mean.max(), x[np.argmax(y_mean)]))


def join_curves(LOG_FOLDER: str, WHICH_LOGS: tuple, X_AXIS_TIMESTEPS: bool) -> tuple:
    """
    Join multiple learning curves into one plot.
    :param LOG_FOLDER (str): folder containing log files of a given baselines environment.
    :param WHICH_LOGS (tuple): tuple of strings containing the names of the log files to be joined and the legend label.
    :param X_AXIS_TIMESTEPS (bool): if True, x-axis is timesteps, else x-axis is episodes.
    :return (tuple): tuple containing the x-axis values (timesteps or episodes) and the y-axis values (rewards).
    """
    i = 0
    for log, title in WHICH_LOGS:
        results = pu.load_results(LOG_FOLDER + log)
        r = results[0]
        steps = (
            np.cumsum(r.monitor.l) if X_AXIS_TIMESTEPS else np.array(r.monitor.l.index)
        )
        rewards = r.monitor.r
        if i == 0:
            if X_AXIS_TIMESTEPS:
                steps_joined = steps.values
            else:
                steps_joined = steps
            rewards_joined = rewards
        else:
            rewards.index = rewards.index + len(rewards_joined)
            rewards_joined = pd.concat([rewards_joined, rewards])
            if X_AXIS_TIMESTEPS:
                steps_joined = np.concatenate(
                    (steps_joined, steps.values + steps_joined[-1])
                )
            else:
                steps_joined = np.concatenate((steps_joined, steps + steps_joined[-1]))
        i += 1
    return steps_joined, rewards_joined


def unpack_dict_parameters(rl_model_parameters_dict: dict) -> tuple:
    """
    Unpack dictionary of parameters into a tuple.
    :param pars_dictionary (dict): dictionary of model parameters.
    :return (tuple): tuple containing the parameters.
    """
    try:
        rl_algo = rl_model_parameters_dict["rl_algo"]
        assert type(rl_algo) == str
    except KeyError:
        rl_algo = "ppo2"

    try:
        training_timesteps = rl_model_parameters_dict["training_timesteps"]
        assert type(training_timesteps) == str
    except KeyError:
        raise KeyError("training_timesteps not found in dictionary of parameters.")

    try:
        number_nn_layers = rl_model_parameters_dict["num_layers"]
        assert type(number_nn_layers) == str
    except KeyError:
        raise KeyError("num_layers not found in dictionary of parameters.")

    try:
        number_nn_units = rl_model_parameters_dict["num_hidden"]
        assert type(number_nn_units) == str
    except KeyError:
        raise KeyError("num_hidden not found in dictionary of parameters.")

    try:
        learning_rate = rl_model_parameters_dict["learning_rate"]
        assert type(learning_rate) == str
    except KeyError:
        raise KeyError("learning_rate not found in dictionary of parameters.")

    try:
        training_seed = rl_model_parameters_dict["training_seed"]
        assert type(training_seed) == str
    except KeyError:
        training_seed = None

    try:
        activation = rl_model_parameters_dict["activation"]
        assert type(activation) == str
    except KeyError:
        activation = None

    try:
        value_network = rl_model_parameters_dict["value_network"]
        assert type(value_network) == str or value_network == None
    except KeyError:
        value_network = None

    try:
        beta = rl_model_parameters_dict["beta"]
        assert type(beta) == str
    except KeyError:
        beta = None

    try:
        entropy_coeff = rl_model_parameters_dict["ent"]
        assert type(entropy_coeff) == str
    except KeyError:
        entropy_coeff = None

    try:
        lam = rl_model_parameters_dict["lam"]
        assert type(entropy_coeff) == str
    except KeyError:
        lam = None

    try:
        noise = rl_model_parameters_dict["noise"]
        assert type(noise) == str
    except KeyError:
        noise = None

    try:
        gamma = rl_model_parameters_dict["gamma"]
        assert type(gamma) == str
    except KeyError:
        gamma = None

    try:
        batch = rl_model_parameters_dict["batch"]
        assert type(batch) == str
    except KeyError:
        batch = None

    try:
        custom_suffix = rl_model_parameters_dict["custom_suffix"]
        assert type(custom_suffix) == str
    except KeyError:
        custom_suffix = ""

    try:
        env_seed = rl_model_parameters_dict["env_seed"]
        assert type(env_seed) in [str, int]
    except KeyError:
        env_seed = "14"

    return (
        rl_algo,
        training_timesteps,
        number_nn_layers,
        number_nn_units,
        learning_rate,
        training_seed,
        activation,
        value_network,
        beta,
        entropy_coeff,
        lam,
        noise,
        batch,
        gamma,
        custom_suffix,
        env_seed,
    )


def test_build_args(
    env_id: str,
    alg: str,
    training_timesteps: str,
    num_layers: str,
    num_hidden: str,
    lr: str,
    training_seed: str = None,
    activation: str = None,
    value_network: str = None,
    beta: str = None,
    ent: str = None,
    custom_suffix: str = None,
    lam: str = None,
    noise: str = None,
    batch: str = None,
    gamma: str = None,
    env_seed: int = 14,
) -> list:
    """
    Utility function for the most common argument configurations.
    :param env_id (str): environment id.
    :param alg (str): reinforcement learning algorithm name.
    :param training_timesteps (int in string format): number of training timesteps used for the model.
    :param num_layers (int in string format): number of hidden layers of the neural network model.
    :param num_hidden (in in string format): number of hidden units in each hidden layer of the neural network model.
    :param lr (float in in string format): learning rate of training algorithm.
    :param activation (str): activation function of the neural network model.
    :param value_network(str, default=None): "None/shared": creates one NN for policy and value function; "copy": creates two different NN for policy and value function.
    :param beta (float in string format, default=None): (default 0.5) importance of value function in the overall loss function. This is a ppo specific hyper-parameter (vf_coef).
    :param ent (float in string format, default=None): (default 0.0) importance of the entropy member in the overall loss function. This is a ppo specific hyper-parameter (ent_coef).
    :param custom_suffix (str, default=None): custom suffix for the log file name.
    :param env_seed (int, default=14): random seed for the environment.
    :return (list): list of arguments for the training script.
    """

    if custom_suffix is not None and custom_suffix != "" and custom_suffix[0] != "_":
        custom_suffix = "_" + custom_suffix

    suffix = "" if custom_suffix is None else custom_suffix

    if activation is not None:
        suffix = "_" + activation + suffix

    if value_network is not None:
        suffix = "_" + value_network + suffix

    if lam is not None:
        suffix = "_lam" + lam + suffix

    if noise is not None:
        suffix = "_noise" + noise + suffix

    if beta is not None:
        suffix = "_beta" + beta + suffix

    if ent is not None:
        suffix = "_ent" + ent + suffix

    if batch is not None:
        suffix = "_batch" + batch + suffix

    if gamma is not None:
        suffix = "_gamma" + gamma + suffix

    if training_seed is not None:
        suffix = "_trainingseed" + training_seed + suffix

    description = "{}_{}_{}x{}_{}{}".format(
        alg, training_timesteps, num_layers, num_hidden, lr, suffix
    )
    args = [
        "--env={}".format(env_id),
        "--num_env=1",
        "--num_timesteps=1",
        "--num_layers={}".format(num_layers),
        "--num_hidden={}".format(num_hidden),
        "--alg={}".format(alg),
        "--lr={}".format(lr),
        "--activation=tf.nn.{}".format(activation),
        "--load_path=./trained_agents/{}/{}".format(env_id, description),
        "--seed={}".format(env_seed),
    ]
    if value_network:
        args.append("--value_network={}".format(value_network))

    if beta:
        args.append("--vf_coef={}".format(beta))
    if ent:
        args.append("--ent_coef={}".format(ent))

    return args


def plot_one_episode_actions(env_id: str, rl_model_parameters_dict: dict) -> None:
    """
    Plots the actions taken by the agent in one episode.
    :param env_id (str): environment id.
    :param rl_model_parameters_dict (dict): dictionary containing the parameters of the RL model to load.
    """
    (
        rl_algorithm,
        training_timesteps,
        number_nn_layers,
        number_nn_units,
        learning_rate,
        training_seed,
        activation_function,
        value_network,
        beta,
        entropy_coeff,
        lam,
        noise,
        batch,
        gamma,
        custom_suffix,
        env_seed,
    ) = unpack_dict_parameters(rl_model_parameters_dict)
    args = test_build_args(
        env_id,
        rl_algorithm,
        training_timesteps,
        number_nn_layers,
        number_nn_units,
        learning_rate,
        training_seed,
        activation_function,
        value_network,
        beta,
        entropy_coeff,
        custom_suffix,
        lam,
        noise,
        batch,
        gamma,
        env_seed,
    )
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    # create environment and agent
    model, env = train(args, extra_args)
    remove_train_noise(model)  # make the policy deterministic for the test session
    # Simulate one episode
    observation = env.reset()
    action_history = np.array([])
    time_grid = np.array([observation[0, -1]])
    done = False
    step = 0
    while done != 1:
        action, values, _, _ = model.step(observation[0], stochastic=False)
        if step == 0:
            action_history = action[0]
        else:
            action_history = np.vstack([action_history, action[0]])
        observation, reward, done, _ = env.step(action[0])
        time_grid = np.append(time_grid, observation[0, -1])
        step += 1
        done = done[0]
    time_grid = time_grid[:-1]
    # Plot the agent actions
    n_assets = len(action_history[0])
    labels_font_size = 13
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    for i in range(n_assets):
        plt.step(
            time_grid, action_history[:, i], label="Asset " + str(i + 1), where="post"
        )
    plt.legend(prop={"size": labels_font_size})
    plt.xlabel("Time [yr]", fontsize=labels_font_size)
    plt.ylabel("Action", fontsize=labels_font_size)
    plt.title("RL agent actions", fontsize=labels_font_size)
    plt.show()


def plot_action_space(
    env_id: str,
    rl_model_parameters_dict: dict,
    plot_value_function: bool,
    reference_state: np.ndarray,
    variable_indexes: list or tuple,
    variable_points: int,
    x_max=np.infty,
    y_max=np.infty,
):
    """
    Function to plot a section of the action space or the value function of the RL agent (2d or 3d plot).
    :param env_id (str): environment id.
    :param rl_model_parameters_dict (dict): dictionary containing the parameters of the RL model to load.
    :param plot_value_function (bool): if True, plots the value function of the RL agent else the action space.
    :param reference_state (np.ndarray): reference state used to plot the action space.
    :param variable_indexes (list or tuple): indexes of the variables to plot.
    :param variable_points (int): number of points to plot in each variable.
    :param x_max (float, default=np.infty): maximum value of the x variable.
    :param y_max (float, default=np.infty): maximum value of the y variable.
    """

    from gym import spaces
    from matplotlib import cm

    # Set-up the title of the plot
    state_space_description = r"$[log(S/F(0,T))\; , I_t/I_0\; , t]$"
    reference_str = list(map(str, reference_state))
    for ivar in variable_indexes:
        reference_str[ivar] = ":"
    z_name = "Value function" if plot_value_function else "Actions"
    title_plot = (
        "{} at {} = [".format(z_name, state_space_description)
        + ", ".join(reference_str)
        + "]"
    )

    labels_font_size = 15  # font-size for the plot
    # fundamental switch
    assert type(variable_indexes) in [list, tuple]
    if len(variable_indexes) == 1:
        plot3d = False
    elif len(variable_indexes) == 2:
        plot3d = True
    else:
        raise ValueError("Either 1 or 2 variables can vary in plot()")

    # selected code from baselines.run.main()
    (
        rl_algorithm,
        training_timesteps,
        number_nn_layers,
        number_nn_units,
        learning_rate,
        training_seed,
        activation_function,
        value_network,
        beta,
        entropy_coeff,
        lam,
        noise,
        batch,
        gamma,
        custom_suffix,
        env_seed,
    ) = unpack_dict_parameters(rl_model_parameters_dict)
    args = test_build_args(
        env_id,
        rl_algorithm,
        training_timesteps,
        number_nn_layers,
        number_nn_units,
        learning_rate,
        training_seed,
        activation_function,
        value_network,
        beta,
        entropy_coeff,
        custom_suffix,
        lam,
        noise,
        batch,
        gamma,
        env_seed,
    )

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    # create environment and agent
    model, env = train(args, extra_args)
    remove_train_noise(model)

    # plot coordinates
    x_low = max(env.observation_space.low[variable_indexes[0]], -x_max)
    x_high = min(env.observation_space.high[variable_indexes[0]], x_max)
    x_axis = np.linspace(x_low, x_high, variable_points)
    if plot3d:
        y_low = max(env.observation_space.low[variable_indexes[1]], -y_max)
        y_high = min(env.observation_space.high[variable_indexes[1]], y_max)
        y_axis = np.linspace(y_low, y_high, variable_points)
    else:
        y_axis = []

    z_shape = (variable_points,) * len(variable_indexes)
    if not plot_value_function:
        action_space = env.action_space.shape
        if action_space == ():
            action_space = (1,)
        z_shape += action_space

    z_axis = np.empty(z_shape)

    def get_z(scalar_observation):
        actions, values, _, _ = model.step(scalar_observation, stochastic=False)
        action = np.nan_to_num(actions[0])
        if not isinstance(env.action_space, spaces.Discrete):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        return values[0] if plot_value_function else action

    # init
    obs = reference_state[np.newaxis, :]

    if plot3d:  # plot surfaces
        # compute actions or values
        x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
        for i in range(x_mesh.shape[0]):
            for j in range(x_mesh.shape[1]):
                obs[0, variable_indexes[0]] = x_mesh[i, j]
                obs[0, variable_indexes[1]] = y_mesh[i, j]
                if plot_value_function:  # if value function is requested
                    z_axis[i, j] = get_z(obs)
                else:
                    z_axis[i, j, :] = get_z(obs)
        # plot
        fig = plt.figure(figsize=(15, 8))
        ax = plt.axes(projection="3d")
        if plot_value_function:
            surf = ax.plot_surface(x_mesh, y_mesh, z_axis, cmap=cm.coolwarm)
            fig.colorbar(surf, shrink=0.5, aspect=5)
        else:
            for i in range(action_space[0]):
                surf = ax.plot_surface(
                    x_mesh, y_mesh, z_axis[:, :, i], label="asset {}".format(i + 1)
                )
                surf._edgecolors2d = surf._edgecolor3d  # elemnts to display the legend
                surf._facecolors2d = surf._facecolor3d
            ax.legend(prop={"size": labels_font_size})
        ax.set_xlabel(
            "state[{}]".format(variable_indexes[0]),
            fontsize=labels_font_size,
            labelpad=labels_font_size - 3,
        )
        ax.set_ylabel(
            "state[{}]".format(variable_indexes[1]),
            fontsize=labels_font_size,
            labelpad=labels_font_size - 3,
        )
        ax.set_zlabel(
            "value" if plot_value_function else "action",
            fontsize=labels_font_size,
            labelpad=labels_font_size - 3,
        )
    else:  # plot 1d graph
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # compute actions or values
        for i, x in enumerate(x_axis):
            obs[0][variable_indexes[0]] = x
            z_axis[i] = get_z(obs)
        # plot
        if plot_value_function:
            plt.plot(x_axis, z_axis)
        else:
            for i in range(action_space[0]):
                plt.step(
                    x_axis, z_axis[:, i], label="asset {}".format(i + 1), where="post"
                )
            plt.legend(prop={"size": labels_font_size})
        plt.xlabel("state[{}]".format(variable_indexes[0]), fontsize=labels_font_size)
        plt.ylabel(
            "value" if plot_value_function else "action", fontsize=labels_font_size
        )

    ax.tick_params(axis="both", which="major", labelsize=labels_font_size - 2)
    ax.tick_params(axis="both", which="minor", labelsize=labels_font_size - 2)
    plt.title(title_plot, fontsize=labels_font_size)
    plt.show()
