from baselines.common.cmd_util import common_arg_parser
from baselines.run import parse_cmdline_kwargs, train, remove_train_noise
import numpy as np
from numpy import exp, log, sqrt
from matplotlib import pyplot as plt
from envs.pricing.pricing import EquityForwardCurve, Black, ForwardVariance, DiscountingCurve
from envs.pricing.fake_market import load_fake_market
from envs.pricing.n_sphere import n_sphere_to_cartesian
from envs.pricing.targetvol import CholeskyTDependent
from gym import spaces
import envs.fe_envs
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def model_creation(seed, fixings, n, normalized):
    N_equity = n
    r = 1/100
    maturity=1
    load_fake_market(N_equity, r, maturity)
    D, F, V, correlation, spot_prices = load_fake_market(N_equity, r, maturity)
    model = Black(variance=V,forward_curve = F)
    nu = CholeskyTDependent(V,correlation)
    vola_t = sqrt(np.sum(nu(0)**2,axis=0))
    for time in fixings[1:]:
        vola_t =  np.vstack([vola_t, sqrt(np.sum(nu(time)**2,axis=0))]) 

    b = Black(forward_curve=F, variance=V)
    np.random.seed(seed)
    gen = np.random
    if normalized ==1:
        return log(b.simulate(fixings=fixings, random_gen=gen, corr=correlation)[0]/(spot_prices))/vola_t
    else:
        return b.simulate(fixings=fixings, random_gen=gen, corr=correlation)[0]

    
    

def get_y(scalar_observation, model, env, plot_value, strategy):
        actions, values, _, _ = model.step(scalar_observation, stochastic=False)
        action = np.nan_to_num(actions[0])
        if not isinstance(env.action_space, spaces.Discrete):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        if strategy:
            action = n_sphere_to_cartesian(1,action)**2
        return values[0] if plot_value else action



def plot(args, plot_value, env_map, reference_state, variable_indexes,
         variable_points, title_plot, legend=None, x_max=np.infty, y_max=np.infty, strategy_long=True, all_time_dep = True, seed=10, N_equity=2, normalized=0):
    """Function to plot a section of the action space."""
    # selected code from baselines.run.main()
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
    y_shape = (variable_points,) * len(variable_indexes)
    if strategy_long:
        dim, = env.action_space.shape
        dim+=1
        y_shape+= dim,
    else:
        dim, = env.action_space.shape
        y_shape+= dim,
    y_axis = np.empty(y_shape)
    label = extra_args['load_path'] if legend is None else legend
    # init
    if all_time_dep:
        S = model_creation(seed, x_axis, N_equity, normalized)
        obs = np.insert(S, dim, x_axis, axis=1)
    else:
        obs = reference_state[np.newaxis, :]
    # compute actions or values
    for i, x in enumerate(x_axis):
        if not all_time_dep:
            obs[0][variable_indexes[0]] = x
            o = obs
        else:
            o = obs[i]
        y_axis[i] = get_y(o,model,env,plot_value,strategy_long)
    
    if not plot_value:
        for i in range(dim):
            plt.step(x_axis, y_axis.T[i], label="Equity"+str(i+1),where='post')
        plt.ylabel(r"Strategy $\alpha(t)$")
        plt.title("Action space at [log($\mathbf{S_t/S_0}$),t]")
        plt.legend()

    else:
        plt.plot(x_axis, y_axis[:,0])
        plt.title("Value function")
        plt.ylabel("$V(t)$")
    plt.xlabel("Time [yr]")
    if not all_time_dep:    
        plt.title(title_plot)


def build_args(env, alg, train_timesteps, num_layers, num_hidden, lr, activation=None,
               value_network=None, noise=None, beta=None, ent=None, custom_suffix=None):
    """Utility function for the most common argument configurations."""

    suffix = "" if custom_suffix is None else custom_suffix
    if activation:
        activation = activation
    else:
        activation = "tanh"

    description = '{}_{}_{}x{}_{}{}'.format(alg, train_timesteps, num_layers, num_hidden, lr, suffix)
    args = [
        '--env={}'.format(env),
        '--num_env=1',
        '--num_timesteps=1',  # sic!
        '--num_layers={}'.format(num_layers),
        '--num_hidden={}'.format(num_hidden),
        '--alg={}'.format(alg),
        '--lr={}'.format(lr),
        '--activation=tf.nn.{}'.format(activation),
        '--load_path=./trained_agents/{}/{}'.format(env, description)
    ]
    if value_network:
        args.append('--value_network={}'.format(value_network))
    
    if beta:
        args.append('--vf_coef={}'.format(beta))
    if noise:
        args.append('--init_logstd={}'.format(noise))
              
    return args



def plot3d(args, plot_value, env_map, reference_state, variable_indexes,
         variable_points, title_plot, legend=None, x_max=np.infty, y_max=np.infty, strategy_long=True, N_equity=2):
    """Function to plot a section of the action space."""
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
    y_low = max(env.observation_space.low[variable_indexes[1]], -y_max)
    y_high = min(env.observation_space.high[variable_indexes[1]], y_max)
    y_axis = np.linspace(y_low, y_high, variable_points)
    z_shape = (variable_points,) * len(variable_indexes)  # + env.action_space.shape
    if not plot_value:
        if strategy_long:
            dim, = env.action_space.shape
            dim+=1
            z_shape+= dim,
        else:
            dim, = env.action_space.shape
            z_shape+= dim,
    z_axis = np.empty(z_shape)
    label = extra_args['load_path'] if legend is None else legend

    def get_z(scalar_observation):
        actions, values, _, _ = model.step(scalar_observation, stochastic=False)
        action = np.nan_to_num(actions[0])
        if not isinstance(env.action_space, spaces.Discrete):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        if strategy_long:
            action = n_sphere_to_cartesian(1,action)**2
        return values[0] if plot_value else action

    # init
    obs = reference_state[np.newaxis, :]

    # compute actions or values
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
    for i in range(x_mesh.shape[0]):
        for j in range(x_mesh.shape[1]):
            obs[0][variable_indexes[0]] = x_mesh[i][j]
            obs[0][variable_indexes[1]] = y_mesh[i][j]
            if not plot_value:
                z_axis[i,j,:] = get_z(obs)
            else:
                z_axis[i,j] = get_z(obs)
    # plot
    plt.figure(label)
    fig = plt.figure(figsize=(15,8))
    ax = fig.gca(projection='3d')
    if plot_value:
        surf=ax.plot_surface(x_mesh, y_mesh, z_axis, cmap=cm.coolwarm)
        fig.colorbar(surf, shrink=0.5, aspect=5)
    else:
        for i in range(N_equity):
            surf=ax.plot_surface(x_mesh, y_mesh, z_axis[:,:,i],label="Equity "+str(i+1))
            surf._facecolors2d=surf._facecolors3d
            surf._edgecolors2d=surf._edgecolors3d
      
    ax.set_xlabel('state[{}]'.format(variable_indexes[0]))
    ax.set_ylabel('state[{}]'.format(variable_indexes[1]))
    ax.set_zlabel('value' if plot_value else 'action')
    plt.title(title_plot)
    ax.legend()
    return ax
