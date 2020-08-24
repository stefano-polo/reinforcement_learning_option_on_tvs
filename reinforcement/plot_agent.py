from baselines.common.cmd_util import common_arg_parser
from baselines.run import parse_cmdline_kwargs, train, remove_train_noise
import numpy as np
from numpy import exp
from matplotlib import pyplot as plt
from envs.pricing.pricing import EquityForwardCurve, Black, ForwardVariance, DiscountingCurve
from envs.pricing.n_sphere import n_sphere_to_cartesian
from gym import spaces
import envs.fe_envs


def get_y(scalar_observation, model, env, plot_value, strategy):
        actions, values, _, _ = model.step(scalar_observation, stochastic=False)
        action = np.nan_to_num(actions[0])
        if not isinstance(env.action_space, spaces.Discrete):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        if strategy:
            action = n_sphere_to_cartesian(1,action)**2
        return values[0] if plot_value else action


def model_creation(seed, fixings):
    N_equity = 2
    r = 1/100
    reference=0
    t = reference
    correlation = np.array(([1,0.5],[0.5,1]))
    spot_prices = np.array([100,200,89])
    T_max = 1
    r_t = np.array([r,r,r,r])
    T_discounts = np.array([0.,3/12,4/12.,T_max])      #data observation of the market discounts factor
    market_discounts = exp(-r_t*T_discounts)       #market discounts factor

    T_repo1 = np.array([1/12,4./12,T_max])       #data observation of the market repo rates for equity 1
    repo_rate1 = np.array([0.72,0.42,0.52])/10  #market repo rates for equity 1
    T_repo2 = np.array([1/12.,4/12.,T_max])
    repo_rate2 = np.array([0.22,0.22,0.22])/10
    T_repo3 = np.array([2/12.,5/12.,T_max])
    repo_rate3 = np.array([0.32,0.32,0.12])/10


    sigma1 = np.array([20,20.,20.])/100
    T_sigma1 = np.array([2/12,5./12,T_max])
    K1 = np.array([spot_prices[0],500])
    spot_vola1 = np.array((sigma1,sigma1))                                      #market implied volatility matrix
    sigma2 = np.array([20,20,20])/100
    T_sigma2 =  np.array([2/12.,6/12,T_max])
    K2 = np.array([spot_prices[1],600])
    spot_vola2 = np.array((sigma2,sigma2))

    D = DiscountingCurve(reference=t, discounts=market_discounts,dates=T_discounts)
    F = []
    V = []
    q = repo_rate1
    T_q = T_repo1
    s_vola = spot_vola1
    T_vola = T_sigma1
    K = K1
    F.append(EquityForwardCurve(reference=0,spot=spot_prices[0],discounting_curve=D,repo_dates=T_q,repo_rates=q))
    V.append(ForwardVariance(reference=0,spot_volatility=s_vola,strikes=K,maturities=T_vola,strike_interp=spot_prices[0]))
    q = repo_rate2
    T_q = T_repo2
    s_vola = spot_vola2
    T_vola = T_sigma2
    K = K2
    F.append(EquityForwardCurve(reference=0,spot=spot_prices[1],discounting_curve=D,repo_dates=T_q,repo_rates=q))
    V.append(ForwardVariance(reference=0,spot_volatility=s_vola,strikes=K,maturities=T_vola,strike_interp=spot_prices[1]))
    b = Black(forward_curve=F, variance=V)
    np.random.seed(seed)
    gen = np.random
    return b.simulate(fixings=fixings, random_gen=gen, corr=correlation)[0]



def plot(args, plot_value, env_map, reference_state, variable_indexes,
         variable_points, title_plot, legend=None, x_max=np.infty, y_max=np.infty, strategy_long=True, all_time_dep = True):
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
    if not plot_value:
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
        S = model_creation(10, x_axis)
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
    print(y_axis)
    if not plot_value:
        for i in range(dim):
            plt.plot(x_axis, y_axis.T[i], label="Equity"+str(i+1))
    else:
        plt.plot(x_axis, y_axis)
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

    return args
