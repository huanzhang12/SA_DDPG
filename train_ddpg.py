#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *
import sys
from argparser import argparser
from config import load_config
from utils import save_runtime

from robust_ddpg import RobustDDPGAgent, RobustDeterministicActorCriticNet
import torch


# Directly concat state and action in first layer
class SimpleTwoLayerFCBodyWithAction(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
        super(SimpleTwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim + action_dim, hidden_size1))
        self.fc2 = layer_init(nn.Linear(hidden_size1, hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        z = self.gate(self.fc1(torch.cat([x, action], dim=1)))
        phi = self.gate(self.fc2(z))
        return phi

# config_dict is our configurations set in JSON file
def ddpg_continuous(config_dict):
    # Read config file and translate relevant fields
    ddpg_config = {}
    ddpg_config['game'] = config_dict['env_id']
    training_config = config_dict['training_config']
    buffer_params = training_config['buffer_params']
    ddpg_config['mini_batch_size'] = training_config['batch_size']

    generate_tag(ddpg_config)
    ddpg_config.setdefault('log_level', 0)
    config = Config()
    config.merge(ddpg_config)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = training_config['num_frames']
    config.eval_interval = training_config['eval_interval']
    config.save_interval = training_config["save_frame"]
    config.eval_episodes = training_config['eval_episodes']
    config.mini_batch_size = training_config['batch_size']
    # newly added properties
    config.robust_params = training_config['robust_params']
    # Setting beta scheduler to match the parameters of epsilon scheuler
    if config.robust_params['beta_scheduler']['start_step'] == 'auto':
        config.robust_params['beta_scheduler']['start_step'] = config.robust_params['eps_scheduler']['start_step']
        print('robust_params[beta_scheduler][start_step] is setting to', config.robust_params['beta_scheduler']['start_step'])
    if config.robust_params['beta_scheduler']['steps'] == 'auto':
        config.robust_params['beta_scheduler']['steps'] = config.robust_params['eps_scheduler']['steps']
        print('robust_params[beta_scheduler][steps] is setting to', config.robust_params['beta_scheduler']['steps'])
    # Setting adv training scheduler eps to match the parameters of epsilon scheuler
    if config.robust_params['advtrain_scheduler']['start'] == 'auto':
        config.robust_params['advtrain_scheduler']['start'] = config.robust_params['eps_scheduler']['end']
        print('robust_params[advtrain_scheduler][start] is setting to', config.robust_params['advtrain_scheduler']['start'])
    if config.robust_params['advtrain_scheduler']['end'] == 'auto':
        config.robust_params['advtrain_scheduler']['end'] = config.robust_params['eps_scheduler']['end']
        print('robust_params[advtrain_scheduler][end] is setting to', config.robust_params['advtrain_scheduler']['end'])
    config.ddpg_debug = {"print_frame": training_config['print_frame'], "profile_time": training_config['profile_time']}
    config.models_path = config_dict["models_path"]
    config.show_game = training_config["show_game"]
    config.load_pretrain = training_config["pretrain_path"]
    config.actor_network = training_config["actor_network"]
    config.critic_network = training_config["critic_network"]
    config.data_params = config_dict["data_config"]

    config.reward_normalizer = RescaleNormalizer(coef=training_config["reward_scaling"])

    config.network_fn = lambda: RobustDeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        config.actor_network, config.critic_network,
        config.mini_batch_size,
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=training_config['actor_lr']),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=training_config['critic_lr']),
        robust_params=config.robust_params)

    config.replay_fn = lambda: MemoryEfficientReplay(memory_size=buffer_params['buffer_capacity'], batch_size=config.mini_batch_size)
    config.discount = training_config["gamma"]
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    config.warm_up = buffer_params['replay_initial']
    config.target_network_mix = training_config['target_network_mix']

    sys.stdout.flush()
    sys.stderr.flush()
    agent = RobustDDPGAgent(config)
    logger = agent.logger
    # Load pretrained model
    if training_config["pretrain_path"]:
        logger.info('Loading %s', training_config["pretrain_path"])
        agent.load(training_config["pretrain_path"])
    # Save all runtime information (command line, json, git commit, etc)
    save_runtime(agent, config_dict["models_path"])
    logger.info('Training started!\n %s', str(config.__dict__))
    run_steps(agent)
    logger.info('[END-END-END] Finishing training')


def main(args):
    config = load_config(args)
    # game_list = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Swimmer-v2', 'Reacher-v2', 'InvertedPendulum-v2', 'Ant-v2', 'Humanoid-v2']
    config['models_path'] = os.path.join(args.path_prefix, config['models_path'].format(env_id=config['env_id']))
    mkdir(config['models_path'])
    mkdir(os.path.join(config['models_path'], 'checkpoints'))
    mkdir(os.path.join(config['models_path'], 'log'))
    mkdir(os.path.join(config['models_path'], 'tf_log'))
    mkdir(os.path.join(config['models_path'], 'runtime'))
    if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
        # Use CPU.
        del os.environ['CUDA_VISIBLE_DEVICES'] # Avoid HIP error.
        if 'HIP_VISIBLE_DEVICES' in os.environ:
            del os.environ['HIP_VISIBLE_DEVICES'] # Avoid HIP error.
        print('Using CPU.')
        select_device(-1)
    else:
        print('Using GPU.')
        select_device(0)
    ddpg_continuous(config)

if __name__ == "__main__":
    args = argparser()
    main(args)
