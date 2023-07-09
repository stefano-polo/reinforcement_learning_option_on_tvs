import sys

sys.path.insert(1, "./src")

import reinforcement.envs.fe_envs
from baselines.run import main


def build_args(
    do_train,
    do_test,
    env,
    alg,
    num_layers,
    num_hidden,
    num_env,
    lr,
    train_timesteps,
    test_episodes,
    print_episodes,
    print_period,
    activation=None,
    save_interval=None,
    value_network=None,
    lam=None,
    gamma=None,
    noise=None,
    beta=None,
    ent=None,
    training_seed=None,
    test_seed=None,
    restart_training=None,
    initial_guess=None,
    batch=None,
    custom_options=None,
    custom_suffix=None,
):
    """
    Utility function for the most common argument configurations.
    :param do_train (bool): whether to train the agent.
    :param do_test (bool): whether to test the agent by means of a Monte Carlo simulation. The script uses the saved model built with the same arguments provided in build_args function in the do_train fase.
    :param env (str): environment name (available: TVS_BS-v0, TVS_LV-v0, TVS_LV-v2, and VanillaOption-v0; see the file fe_envs.py).
    :param alg (str): Reinforcement Learning algorithm in baselines format name (example : "ppo2").
    :param num_layers (int in string format): number of hidden layers for the policy neural network (NN).
    :param num_hidden (int in string format): number of neurons for each hidden layer. Thus the built NN has num_layers layers with num_hidden neurons each.
    :param num_env (int in string format): number of environments to run in parallel.
    :param lr (float in string format): (default 3e-4) learning rate.
    :param train_timesteps (int in string format): number of timesteps (episode_lenght * number_of_training_episodes) to train the agent.
    :param test_episodes (int in string format): number of episodes to test the agent.
    :param print_episodes (int in string format): print the results every print_episodes steps.
    :param print_period (int in string format): print the results every print_period steps.
    :param activation (str): activation function for the hidden layers of the NN (the output layer has linear activation function).
    :param save_interval (int in string format): save the model every save_interval steps during the training phase (save_interval is measured in epochs called in baselines as 'nupdates').
    :param value_network (str): "None/shared": creates one NN for policy and value function; "copy": creates two different NN for policy and value function.
    :param lam (float in string format): (default 0.95) parameter that controls the bias and variance of the expected reward estimator (useful for convergence). This is a ppo specific hyper-parameter.
    :param gamma (float in string format): (default 0.99) discount factor of the Reinforcement Learning reward.
    :param noise (float in string format): value for the log(sigma), where sigma is the noise size over the output of the policy NN (action is not deterministic). The noise parameter is set to 1 during the test phase (deterministic policy).
    :param beta (float in string format): (default 0.5) importance of value function in the overall loss function. This is a ppo specific hyper-parameter (vf_coef).
    :param ent (float in string format): (default 0.0) importance of the entropy member in the overall loss function. This is a ppo specific hyper-parameter (ent_coef).
    :param training_seed (int in string format): seed for the random number generator used during the training phase.
    :param test_seed (int in string format): seed for the random number generator used during the test phase.
    :param restart_training (bool): whether to restart the training from a selected saved model (initial_guess).
    :param initial_guess (str): the saved model to restart the training from. The saved model must be located in the folder trained_agents/environment_name.
    :param batch (int in string format): (default 2048) batch size for the training phase.
    :param custom_options (list): custom commands options list (example: '--activation=tf.nn.tanh').
    :param custom_suffix (str): string of characters to add at the end of the folder name.
    """
    if custom_suffix is not None and custom_suffix != "" and custom_suffix[0] != "_":
        custom_suffix = "_" + custom_suffix

    suffix = "" if custom_suffix is None else custom_suffix
    options = [] if custom_options is None else custom_options

    assert do_train != do_test  # "do_train and do_test cannot be both True or False"

    if activation is not None:
        suffix = "_" + activation + suffix
        options.append("--activation=tf.nn." + activation)

    if value_network is not None:
        suffix = "_" + value_network + suffix
        options.append("--value_network=" + value_network)

    if lam is not None:
        suffix = "_lam" + lam + suffix
        options.append("--lam=" + lam)

    if noise is not None:
        suffix = "_noise" + noise + suffix
        if do_train:
            options.append("--init_logstd=" + noise)

    if beta is not None:
        suffix = "_beta" + beta + suffix
        options.append("--vf_coef=" + beta)

    if ent is not None:
        suffix = "_ent" + ent + suffix
        options.append("--ent_coef=" + ent)

    if batch is not None:
        suffix = "_batch" + batch + suffix
        options.append("--nsteps=" + batch)

    if gamma is not None:
        suffix = "_gamma" + gamma + suffix
        if do_train:
            options.append("--gamma=" + gamma)

    if training_seed is not None:
        suffix = "_trainingseed" + training_seed + suffix
        if do_train:
            options.append("--seed=" + training_seed)

    if save_interval is not None:
        if do_train:
            options.append("--save_interval=" + save_interval)

    description = "{}_{}_{}x{}_{}{}".format(
        alg, train_timesteps, num_layers, num_hidden, lr, suffix
    )
    log_path = "./logs/{}/{}".format(env, description)
    agent_path = "./trained_agents/{}/{}".format(env, description)  # saving files

    if do_train:
        agent_mode = "save"
        options.append("--log_path=" + log_path)
        options.append("--lr=" + lr)
        if restart_training:
            if initial_guess is None:
                raise Exception(
                    "You have to provide the initial guess where to start the training"
                )
            initial_guess = "./trained_agents/" + str(env) + "/" + str(initial_guess)
            options.append("--load_path={}".format(initial_guess))
    else:
        train_timesteps = "0"
        agent_mode = "load"

    if do_test:
        options.append("--play_episodes=" + test_episodes)
        options.append("--print_episodes=" + print_episodes)
        options.append("--print_period=" + print_period)
        options.append("--gamma=1.")
        if test_seed is not None:
            options.append("--seed=" + test_seed)

    args = [
        "--env=" + env,
        "--num_env=" + num_env,
        "--num_layers=" + num_layers,
        "--num_hidden=" + num_hidden,
        "--alg=" + alg,
        "--{}_path={}".format(agent_mode, agent_path),
        "--num_timesteps=" + train_timesteps,
    ]
    return args + options


# Run parameters

if __name__ == "__main__":
    cur_args = build_args(
        do_train=True,
        do_test=False,
        restart_training=False,
        test_seed="112",
        training_seed="45891",
        env="TVS_LV-v0",
        alg="ppo2",
        num_layers="5",
        num_hidden="8",
        num_env="30",
        activation="tanh",
        lr="3e-4",
        train_timesteps="1e6",
        test_episodes="1000000",
        print_episodes="1",
        print_period="64",
        save_interval="200",
        value_network="copy",
        beta="0.7",
        custom_suffix="pricing_prova",
    )
    main(cur_args)
