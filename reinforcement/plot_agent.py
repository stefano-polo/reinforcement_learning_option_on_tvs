from baselines.common.cmd_util import common_arg_parser
from baselines.run import parse_cmdline_kwargs, train, remove_train_noise
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from gym import spaces
import envs.fe_envs


def plot(args, plot_value, env_map, reference_state, variable_indexes,
         variable_points, title_plot, legend=None, x_max=np.infty, y_max=np.infty):
    """Function to plot a section of the action space."""

    # fundamental switch
    if len(variable_indexes) == 1:
        plot3d = False
    elif len(variable_indexes) == 2:
        plot3d = True
    else:
        raise ValueError("Either 1 or 2 variables can vary in plot()")

    # selected code from baselines.run.main()
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    # create environment and agent
    model, env = train(args, extra_args, {'env': env_map})
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

    z_shape = (variable_points,) * len(variable_indexes)  # + env.action_space.shape
    z_axis = np.empty(z_shape)
    label = extra_args['load_path'] if legend is None else legend

    def get_z(scalar_observation):
        actions, values, _, _ = model.step(scalar_observation, stochastic=False)
        action = np.nan_to_num(actions[0])
        if not isinstance(env.action_space, spaces.Discrete):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        action = env.envs[0].interpret_action(action)
        return values[0] if plot_value else action

    # init
    obs = reference_state[np.newaxis, :]

    if plot3d:
        # compute actions or values
        x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
        for i in range(x_mesh.shape[0]):
            for j in range(x_mesh.shape[1]):
                obs[0][variable_indexes[0]] = x_mesh[i][j]
                obs[0][variable_indexes[1]] = y_mesh[i][j]
                z_axis[i][j] = get_z(obs)
        # plot
        plt.figure(label)
        ax = plt.axes(projection='3d')
        ax.plot_surface(x_mesh, y_mesh, z_axis)
        ax.set_xlabel('state[{}]'.format(variable_indexes[0]))
        ax.set_ylabel('state[{}]'.format(variable_indexes[1]))
        ax.set_zlabel('value' if plot_value else 'action')
    else:
        # compute actions or values
        for i, x in enumerate(x_axis):
            obs[0][variable_indexes[0]] = x
            z_axis[i] = get_z(obs)
        # plot
        plt.plot(x_axis, z_axis, label=label)

    plt.title(title_plot)


def build_args(env, alg, train_timesteps, num_layers, num_hidden, lr, activation=None,
               value_network=None, noise=None, beta=None, ent=None, custom_suffix=None):
    """Utility function for the most common argument configurations."""

    suffix = "" if custom_suffix is None else custom_suffix
    if activation:
        suffix = "_" + activation + suffix
    else:
        activation = "tanh"

    if value_network:
      suffix = "_" + value_network + suffix

    if noise:
        suffix = "_noise" + noise + suffix
    if beta:
        suffix = "_beta" + beta + suffix
    if ent:
        suffix = "_ent" + beta + suffix

    description = '{}_{}_{}x{}_{}{}'.format(alg, train_timesteps, num_layers, num_hidden, lr, suffix)
    args = [
        '--env={}'.format(env),
        '--num_env=1',
        '--num_timesteps=0',  # sic!
        '--num_layers={}'.format(num_layers),
        '--num_hidden={}'.format(num_hidden),
        '--alg={}'.format(alg),
        '--lr={}'.format(lr),
        '--activation=tf.nn.{}'.format(activation),
        '--load_path=./trained_agents/{}/{}'.format(env, description)
    ]
    if value_network:
        args.append('--value_network={}'.format(value_network))

    return args


# MAIN: plot actions for several agents

# configurations: which environment and agents
PLOT_VALUE = False  # otherwise plots actions
VARIABLE_INDEXES = (0, )  # tuple of length 1 (2d plot) or 2 (3d plot)
VARIABLE_POINTS = 400
ENV = 'CompoundOption-v0'
#REFERENCE_STATE = np.array([0.0, 0.003654795])
REFERENCE_STATE = np.array([0.0, 0.02])
X_MAX = np.infty
ACTION_SPACE_DESCRIPTION = '[log(S/S0), t]'
AGENTS = [
    #(build_args(ENV, 'deepq', '1e8', '3', '4', '3e-4', custom_suffix='_cpepisodes10000_tgtfreq100000_updfreq250_batch7500_grid100'), {}, 'DeepQ'),
    #(build_args(ENV, 'ppo2', '1.6e9', '3', '4', '3e-4', custom_suffix='_lam09'), {'action_grid_size': 0}, 'PPO'),
    (build_args(ENV, 'ppo2', '1e6', '5', '4', '3e-4', value_network='copy', beta='0.01', custom_suffix='PROVA'), {'action_grid_size': 0}, 'PPO'),
    (build_args(ENV, 'custom', '', '', '', ''), {'action_grid_size': 0}, 'B&S delta')
    ]
"""
ENV = 'GemSwingOption-v0'
REFERENCE_STATE = np.array([0.0, 0.0294, 0.097031963])
X_MAX = 0.4
ACTION_SPACE_DESCRIPTION = '[log(S/S0), (C - center) / max, t]'
AGENTS = [
    (build_args(ENV, 'ppo2', '1e8', '5', '4', '1e-3',
                value_network='copy', custom_suffix='_1m_12_20'), False, '5x4'),
    ]
"""

reference_str = list(map(str, REFERENCE_STATE))
for ivar in VARIABLE_INDEXES:
    reference_str[ivar] = ':'
z_name = 'Value function' if PLOT_VALUE else 'Actions'
title = '{} at {} = ['.format(z_name, ACTION_SPACE_DESCRIPTION) + ', '.join(reference_str) + ']'

for (arg, env_args, lbl) in AGENTS:
    g = tf.Graph()
    sess = tf.InteractiveSession(graph=g)
    with g.as_default():
        plot(arg, PLOT_VALUE, env_args, REFERENCE_STATE, VARIABLE_INDEXES, VARIABLE_POINTS, title, lbl, X_MAX)

plt.legend()
plt.show()
