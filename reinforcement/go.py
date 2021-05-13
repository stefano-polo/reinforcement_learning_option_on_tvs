#! python

from baselines.run import main
import envs.fe_envs

############################################ MANDATORY PARAMETERS #################################################
# do_train (boolean) = if you want to train the agent or to do the MC to obtain the price of the strategy
# do_test (boolean) = if you want to do MC
# env (string) = enviroment name
# alg (string) = rl algorithm (usually ppo2)
# num_layers (string) = number of hidden layers
# num_hidden (string) = number of neurons for each hidden layer
# num_env (string) = number for multiprocessing python (parallelism)
# lr (string) = learning rate (about 3e-4)
# train_timesteps (string) = how much time it spends to train the agent (approximately how many episodes) (about 1e8)
# test_episodes (string) = number of episodes for the monte carlo calculation (you need do_test=True)
# print_episodes (string: 0 or 1) = print some information
# print_period (string: number) = every how many steps it prints a log about the trainign
###################################################################################################################

# value_network (string) = "None/shared": creates one NN for policy and value function; "copy": creates two different NN for policy and value function
# lamb (double) = (default 0.95) parameter that controls the bias and variance of the expected reward estimator (useful for convergence)
# noise (double) = add noise over the output of the NN (action is not deterministic)
# beta (double) = importance of value function
# custom_suffix (string) = add a string of characters at the end of the folder name



def build_args(do_train,do_test, env, alg, num_layers, num_hidden, num_env, lr,
               train_timesteps, test_episodes, print_episodes, print_period, activation=None, save_interval=None,
               value_network=None, lam=None, gamma=None, noise=None, beta=None, ent=None, training_seed=None,
               test_seed=None, restart_training=None, initial_guess=None,
               batch=None, custom_options=None, custom_suffix=None):
    """Utility function for the most common argument configurations."""

    #do_test = not do_train
    suffix = "" if custom_suffix is None else custom_suffix
    options = [] if custom_options is None else custom_options

    if activation is not None:
        suffix = '_' + activation + suffix
        options.append('--activation=tf.nn.' + activation)

    if value_network is not None:
        suffix = '_' + value_network + suffix
        options.append('--value_network=' + value_network)

    if lam is not None:
        suffix = '_lam' + lam + suffix
        options.append('--lam=' + lam)

    if noise is not None:
        suffix = '_noise' + noise + suffix
        if do_train:
            options.append('--init_logstd=' + noise)

    if beta is not None:
        suffix = '_beta' + beta + suffix
        options.append('--vf_coef=' + beta)

    if ent is not None:
        suffix = '_ent' + ent + suffix
        options.append('--ent_coef=' + ent)

    if batch is not None:
        suffix = '_batch' + batch + suffix
        options.append('--nsteps=' + batch)
      
    if gamma is not None:
        suffix = '_gamma' + gamma + suffix
        if do_train:
            options.append('--gamma=' + gamma)

    if training_seed is not None:
        suffix = '_trainingseed' + training_seed + suffix
        if do_train:
            options.append('--seed=' + training_seed)
       
    if save_interval is not None:
        if do_train:
            options.append('--save_interval=' + save_interval)
    
    description = '{}_{}_{}x{}_{}{}'.format(alg, train_timesteps, num_layers, num_hidden, lr, suffix)
    log_path='./logs/{}/{}'.format(env, description)
    agent_path='./trained_agents/{}/{}'.format(env, description)  #saving files

    if do_train:
        agent_mode='save'
        options.append('--log_path=' + log_path)
        options.append('--lr=' + lr)
        if restart_training:
            if initial_guess is None: 
                raise Exception("You have to provide the initial guess where to start the training")
            initial_guess = './trained_agents/'+str(env)+'/'+str(initial_guess)
            options.append('--load_path={}'.format(initial_guess))
    else:
        train_timesteps='0'
        agent_mode='load'

    if do_test:
        options.append('--play_episodes=' + test_episodes)
        options.append('--print_episodes=' + print_episodes)
        options.append('--print_period=' + print_period)
        options.append('--gamma=1.')
        if test_seed is not None:
            options.append('--seed=' + test_seed)

    args = [
        '--env=' + env,
        '--num_env=' + num_env,
        '--num_layers=' + num_layers,
        '--num_hidden=' + num_hidden,
        '--alg=' + alg,
        '--{}_path={}'.format(agent_mode, agent_path),
        '--num_timesteps=' + train_timesteps,
    ]

    return args + options


# Run parameters

if __name__ == '__main__':
    cur_args = build_args(
        do_train=True,
        do_test=False,
        test_seed='114',
        training_seed='34561',
        env='TVS_LV_newreward-v0',
        alg='ppo2',
        num_layers='5',
        num_hidden='8',
        num_env='27',
        lr='3e-4',
        gamma='0.9',  
        train_timesteps='8e7',
        test_episodes='1e6',
        print_episodes='1',
        print_period='64',
        save_interval='200',    
        value_network='copy',
        noise='7.',
        beta='0.7',
        custom_suffix='_freestrategy_displacedmarket_2assets_monthgrid_maturity2_strikeatm'   #test on one_month 6,10,1e6
        )
    main(cur_args)
