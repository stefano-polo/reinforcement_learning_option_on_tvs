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

def model_creation(seed, fixings, n, normalized, market = 0):
    if not market:
        N_equity = n
        print('Loading fake market with number of equity: ',n)
        r = 1/100
        maturity=1
        load_fake_market(N_equity, r, maturity)
        D, F, V, correlation, spot_prices = load_fake_market(N_equity, r, maturity)
        names = []
        for i in range(N_equity):
            names.append("Equity "+str(i+1))
    elif market:
        N_equity = n
        print('Loading real market with number of equity: ',n)
        reader = MarketDataReader("TVS_example.xml")
        correlation = reader.get_correlation()
        names = reader.get_stock_names()
        D = reader.get_discounts()
        F = reader.get_forward_curves()
        spot_prices = reader.get_spot_prices()
        V = reader.get_volatilities()
        if n==2:
            print('Cutting real data')
            F = [F[3],F[4]]
            V = [V[3],V[4]]
            spot_prices = np.array([spot_prices[3],spot_prices[4]])
            correlation = np.array(([1.,0.],[0.,1.]))
            names = [names[3],names[4]]
        
    b = Black(fixings=fixings,forward_curve=F, variance_curve=V)
    nu = CholeskyTDependent(V,correlation)
    chole = np.linalg.cholesky(correlation)
    gen = np.random
    gen.seed(seed)
    simulation = b.simulate(random_gen=gen, corr_chole=chole)[0]
    if normalized ==1:
        vola_t = sqrt(np.sum(nu(fixings)**2,axis=1)).T
        simulation = log(simulation/spot_prices)/vola_t
        print('Normalization with $\sigma(t)$')
        return simulation, names
    elif normalized == 2:
        simulation[1:] = (log(simulation[1:]/simulation[:-1])-0.5*b.variance.T[1:])/sqrt(b.variance.T[1:])
        simulation[0]=0.
        print('Normalization with variance')
        return simulation, names
    else:
        print('No normalization of the input data')
        return simulation,names

    
    

def get_y(scalar_observation, model, env, plot_value, strategy):
        actions, values, _, _ = model.step(scalar_observation, stochastic=False)
        action = np.nan_to_num(actions[0])
        if not isinstance(env.action_space, spaces.Discrete):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        if strategy:
            action = n_sphere_to_cartesian(1,action)**2
        return values[0] if plot_value else action



def plot(args, plot_value, env_map, reference_state, variable_indexes,
         variable_points, title_plot, legend=None, x_max=np.infty, y_max=np.infty, strategy_long=True, all_time_dep = True, seed=10, N_equity=2, normalized=0, market=0):
    """Function to plot a section of the action space."""
    # selected code from baselines.run.main()
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    # create environment and agent
    model, env = train(args, extra_args)
    remove_train_noise(model)
    
    x_low = max(env.observation_space.low[variable_indexes[0]], -x_max)
    x_high = min(env.observation_space.high[variable_indexes[0]], x_max)
    
    x_axis = np.linspace(x_high/variable_points, x_high, variable_points)
    x_axis = np.insert(x_axis,0,x_low)
    print("Evaluation grid ",x_axis)
    y_shape = (variable_points+1,) * len(variable_indexes)
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
        S, names = model_creation(seed, x_axis, N_equity, normalized)
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
            plt.step(x_axis, y_axis.T[i], label=names[i],where='post')
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
