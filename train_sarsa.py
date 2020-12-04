from deep_rl import *
from argparser import argparser
from utils import save_runtime


from sarsa_ddpg import RobustDDPGAgent
from train_ddpg import RobustDeterministicActorCriticNet
import torch

# Special trainer for Sarsa value function used for attacks.
# config_dict is our configurations set in JSON file
def ddpg_continuous(config_dict):
    # Read config file and translate relevant fields
    ddpg_config = {}
    ddpg_config['game'] = config_dict['env_id']
    training_config = config_dict['training_config']
    test_config = config_dict['test_config']
    buffer_params = training_config['buffer_params']
    sarsa_params = test_config["sarsa_params"]
    ddpg_config['mini_batch_size'] = training_config['batch_size']

    generate_tag(ddpg_config)
    ddpg_config.setdefault('log_level', 0)
    config = Config()
    config.merge(ddpg_config)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.test_config = config_dict['test_config']

    # Override several training parameters specially for sarsa.
    config.max_steps = sarsa_params['num_steps'] + sarsa_params['sample_size']
    # We only save at the end.
    config.eval_interval = config.max_steps
    # Never save checkpoint.
    config.save_interval = config.max_steps * 2
    config.warm_up = sarsa_params['sample_size']
    config.sarsa_params = sarsa_params
    
    config.eval_episodes = training_config['eval_episodes']
    config.mini_batch_size = training_config['batch_size']
    config.robust_params = training_config['robust_params']
    config.ddpg_debug = {"print_frame": training_config['print_frame'], "profile_time": training_config['profile_time']}
    config.models_path = config_dict["models_path"]
    config.show_game = training_config["show_game"]
    config.load_pretrain = training_config["pretrain_path"]
    config.actor_network = training_config["actor_network"]
    config.critic_network = training_config["critic_network"]
    config.data_params = config_dict["data_config"]
    config.attack_params = test_config["attack_params"]
    if config.attack_params['alpha'] == "auto":
        config.attack_params['alpha'] = config.attack_params['eps'] / config.attack_params['iteration']

    config.reward_normalizer = RescaleNormalizer(coef=training_config["reward_scaling"])

    config.network_fn = lambda: RobustDeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        config.actor_network, config.critic_network,
        config.mini_batch_size,
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=training_config['actor_lr']),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=training_config['critic_lr']))

    config.replay_fn = lambda: Replay(memory_size=buffer_params['buffer_capacity'], batch_size=config.mini_batch_size)
    config.discount = training_config["gamma"]
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    config.target_network_mix = training_config['target_network_mix']

    agent = RobustDDPGAgent(config)
    logger = agent.logger
   
    # load trained agency
    best_model = os.path.join(agent.config.models_path, "model_best")
    agent.load(best_model)
    agent.logger.info('Loading trained agency %s', best_model)
    # Save all runtime information (command line, json, git commit, etc)
    save_runtime(agent, config_dict["models_path"])
    logger.info('Training sarsa started!\n %s', str(config.__dict__))
    run_steps(agent)
