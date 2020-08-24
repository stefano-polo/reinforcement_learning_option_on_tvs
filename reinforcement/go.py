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



def build_args(do_train, do_test, env, alg, num_layers, num_hidden, num_env, lr,
               train_timesteps, test_episodes, print_episodes, print_period, activation=None,
               value_network=None, lam=None, noise=None, beta=None, ent=None, restart_training=None, initial_guess=None,
               batch=None, custom_options=None, custom_suffix=None):
    """Utility function for the most common argument configurations."""

    suffix = "" if custom_suffix is None else custom_suffix
    options = [] if custom_options is None else custom_options

    if activation is not None:
       custom_suffix = '_' + activation + custom_suffix
       options.append('--activation=tf.nn.' + activation)

    if value_network is not None:
       custom_suffix = '_' + value_network + custom_suffix
       options.append('--value_network=' + value_network)

    if lam is not None:
       custom_suffix = '_lam' + lam
       options.append('--lam=' + lam)

    if noise is not None:
       custom_suffix = '_noise' + noise
       options.append('--init_logstd=' + noise)

    if beta is not None:
       custom_suffix = '_beta' + beta
       options.append('--vf_coef=' + beta)

    if ent is not None:
       custom_suffix = '_ent' + ent
       options.append('--ent_coef=' + ent)

    if batch is not None:
       custom_suffix = '_batch' + batch
       options.append('--nsteps=' + batch)

    description = '{}_{}_{}x{}_{}{}'.format(alg, train_timesteps, num_layers, num_hidden, lr, suffix)
    log_path='./logs/{}/{}'.format(env, description)
    agent_path='./trained_agents/{}/{}'.format(env, description)  #saving files

    if do_train:
       agent_mode='save'
       seed='24816'
       options.append('--log_path=' + log_path)
       options.append('--lr=' + lr)
    else:
       train_timesteps='0'
       agent_mode='load'
       seed='1248'

    if do_test:
       options.append('--play_episodes=' + test_episodes)
       options.append('--print_episodes=' + print_episodes)
       options.append('--print_period=' + print_period)

    if do_train and restart_training:
        if initial_guess is None:
            raise Exception("You have to provide the inizial guess where to start the training")
        initial_guess = './trained_agents/'+str(env)+'/'+str(initial_guess)
        args = [
                '--gamma=1.',
                '--env=' + env,
                '--num_env=' + num_env,
                '--num_layers=' + num_layers,
                '--num_hidden=' + num_hidden,
                '--alg=' + alg,
                '--{}_path={}'.format(agent_mode, agent_path),
                '--load_path={}'.format(initial_guess),
                '--num_timesteps=' + train_timesteps,
                '--seed=' + seed
            ]
    else:
        args = [
            '--gamma=1.',
            '--env=' + env,
            '--num_env=' + num_env,
            '--num_layers=' + num_layers,
            '--num_hidden=' + num_hidden,
            '--alg=' + alg,
            '--{}_path={}'.format(agent_mode, agent_path),
            '--num_timesteps=' + train_timesteps,
            '--seed=' + seed
        ]

    return args + options


# Run parameters

if __name__ == '__main__':

    cur_args = build_args(
       do_train=True,
       do_test=False,
       restart_training = False,
       env='TVS_simple-v0',#'TVS_simple-v0',
       alg='ppo2',
       num_layers='2',
       num_hidden='2',
       num_env='3',
       lr='3e-4',
       train_timesteps='1e7',
       test_episodes='1e5',
       print_episodes='1',
       print_period='64',
       #activation='sigmoid',
       value_network='copy',
       custom_suffix='long_month_observation'   #test on one_month 6,10,1e6
    )

    main(cur_args)
