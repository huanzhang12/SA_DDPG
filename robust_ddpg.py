from deep_rl.network import *
from deep_rl.component import *
from deep_rl.agent.BaseAgent import *
from deep_rl.utils.schedule import LinearSchedule
import random
import torch
import torch.nn as nn
import torchvision

from utils import AverageMeter, MultiTimer
from pdb import set_trace as st

# AutoLiRPA Convex relaxation of neural networks.
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.bound_ops import BoundTanh


def generate_mlp_units(in_dim, neurons, out_dim):
    assert len(neurons) >= 1
    # input layer
    units = [nn.Linear(in_dim, neurons[0])]
    prev = neurons[0]
    # intermediate layers
    for n in neurons[1:]:
        units.append(nn.ReLU())
        units.append(nn.Linear(prev, n))
        prev = n
    # output layer
    units.append(nn.ReLU())
    # Orthogonal layer initialization for last layer
    units.append(layer_init(nn.Linear(neurons[-1], out_dim), 1e-3))
    return units


# Create a simple MLP model
def model_mlp_any(in_dim, neurons, out_dim):
    units = generate_mlp_units(in_dim, neurons, out_dim)
    return nn.Sequential(*units)


class model_mlp_any_with_loss(nn.Module):
    def __init__(self, in_dim, neurons, out_dim):
        super().__init__()
        self.units = generate_mlp_units(in_dim, neurons, out_dim)
        for i, u in enumerate(self.units):
            self.add_module(str(i), u)

    def forward(self, x, y0):
        for u in self.units:
            x = u(x)
        y = torch.tanh(x)
        z = y - y0
        return (z * z).sum(axis=1, keepdim=True)

class RobustDeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_network,
                 critic_network,
                 mini_batch_size,
                 actor_opt_fn,
                 critic_opt_fn,
                 robust_params=None):
        super(RobustDeterministicActorCriticNet, self).__init__()

        if robust_params is None:
            robust_params = {}
        self.use_loss_fusion = robust_params.get('use_loss_fusion', False) # Use loss fusion to reduce complexity for convex relaxation. Default is False.
        self.use_full_backward = robust_params.get('use_full_backward', False)
        if self.use_loss_fusion:
            # Use auto_LiRPA to compute the L2 norm directly.
            self.fc_action = model_mlp_any_with_loss(state_dim, actor_network, action_dim)
            modules = self.fc_action._modules
            # Auto LiRPA wrapper
            self.fc_action = BoundedModule(
                    self.fc_action, (torch.empty(size=(1, state_dim)), torch.empty(size=(1, action_dim))), device=Config.DEVICE)
            # self.fc_action._modules = modules
            for n in self.fc_action.nodes:
                # Find the tanh neuron in computational graph
                if isinstance(n, BoundTanh):
                    self.fc_action_after_tanh = n
                    self.fc_action_pre_tanh = n.inputs[0]
                    break
        else:
            # Fully connected layer with [state_dim, 400, 300, action_dim] neurons and ReLU activation function
            self.fc_action = model_mlp_any(state_dim, actor_network, action_dim)
            # auto_lirpa wrapper
            self.fc_action = BoundedModule(
                    self.fc_action, (torch.empty(size=(1, state_dim)), ), device=Config.DEVICE)

        # Fully connected layer with [state_dim + action_dim, 400, 300, 1]
        self.fc_critic = model_mlp_any(state_dim + action_dim, critic_network, 1)
        # auto_lirpa wrapper
        self.fc_critic = BoundedModule(
                self.fc_critic, (torch.empty(size=(1, state_dim + action_dim)), ), device=Config.DEVICE)

        self.actor_params = self.fc_action.parameters()
        self.critic_params = self.fc_critic.parameters()

        self.actor_opt = actor_opt_fn(self.actor_params)
        self.critic_opt = critic_opt_fn(self.critic_params)
        self.to(Config.DEVICE)
        # Create identity specification matrices
        self.actor_identity = torch.eye(action_dim).repeat(mini_batch_size,1,1).to(Config.DEVICE)
        self.critic_identity = torch.eye(1).repeat(mini_batch_size,1,1).to(Config.DEVICE)
        self.action_dim = action_dim
        self.state_dim = state_dim

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        # Not used, originally this is a feature extraction network
        return tensor(obs)

    def actor(self, phi):
        if self.use_loss_fusion:
            self.fc_action(phi, torch.zeros(size=phi.size()[:1] + (self.action_dim,), device=Config.DEVICE))
            return self.fc_action_after_tanh.forward_value
        else:
            return torch.tanh(self.fc_action(phi, method_opt="forward"))

    # Obtain element-wise lower and upper bounds for actor network through convex relaxations.
    def actor_bound(self, phi_lb, phi_ub, beta=1.0, eps=None, norm=np.inf, upper=True, lower=True, phi = None, center = None):
        if self.use_loss_fusion: # Use loss fusion (not typically enabled)
            assert center is not None
            ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=phi_lb, x_U=phi_ub)
            x = BoundedTensor(phi, ptb)
            val = self.fc_action(x, center.detach())
            ilb, iub = self.fc_action.compute_bounds(IBP=True, method=None)
            if beta > 1e-10:
                clb, cub = self.fc_action.compute_bounds(IBP=False, method="backward", bound_lower=False, bound_upper=True)
                ub = cub * beta + iub * (1.0 - beta)
                return ub
            else:
                return iub
        else:
            assert center is None
            # Invoke auto_LiRPA for convex relaxation.
            ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=phi_lb, x_U=phi_ub)
            x = BoundedTensor(phi, ptb)
            if self.use_full_backward:
                clb, cub = self.fc_action.compute_bounds(x=(x,), IBP=False, method="backward")
                return cub, clb
            else:
                ilb, iub = self.fc_action.compute_bounds(x=(x,), IBP=True, method=None)
                if beta > 1e-10:
                    clb, cub = self.fc_action.compute_bounds(IBP=False, method="backward")
                    ub = cub * beta + iub * (1.0 - beta)
                    lb = clb * beta + ilb * (1.0 - beta)
                    return ub, lb
                else:
                    return iub, ilb


    def critic(self, phi, a):
        return self.fc_critic(torch.cat([phi, a], dim=1), method_opt="forward")

    # Obtain element-wise lower and upper bounds for critic network through convex relaxations.
    def critic_bound(self, phi_lb, phi_ub, a_lb, a_ub, beta=1.0, eps=None, phi=None, action=None, norm=np.inf, upper=True, lower=True):
        x_L = torch.cat([phi_lb, a_lb], dim=1)
        x_U = torch.cat([phi_ub, a_ub], dim=1)
        ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=x_L, x_U=x_U)
        x = BoundedTensor(torch.cat([phi, action], dim=1), ptb)
        ilb, iub = self.fc_critic.compute_bounds(x=(x,), IBP=True, method=None)
        if beta > 1e-10:
            clb, cub = self.fc_critic.compute_bounds(IBP=False, method="backward")
            ub = cub * beta + iub * (1.0 - beta)
            lb = clb * beta + ilb * (1.0 - beta)
            return ub, lb
        else:
            return iub, ilb
        
    def load_state_dict(self, state_dict, strict=True):
        action_dict = OrderedDict()
        critic_dict = OrderedDict()
        for k in state_dict.keys():
            if 'action' in k:
                pos = k.find('.') + 1
                action_dict[k[pos:]] = state_dict[k]
            if 'critic' in k:
                pos = k.find('.') + 1
                critic_dict[k[pos:]] = state_dict[k]
        # loading actor and critic networks separtely. this is requried for auto lirpa.
        self.fc_action.load_state_dict(action_dict)
        self.fc_critic.load_state_dict(critic_dict)

    def state_dict(self):
        # save actor and critic networks separtely. this is requried for auto lirpa.
        action_state_dict = self.fc_action.state_dict()
        critic_state_dict = self.fc_critic.state_dict()
        network_state_dict = OrderedDict()
        for k,v in action_state_dict.items():
            network_state_dict["fc_action."+k] = v
        for k,v in critic_state_dict.items():
            network_state_dict["fc_critic."+k] = v
        return network_state_dict


class RobustDDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        print('origin network keys:', self.network.state_dict().keys())
        self.target_network = config.network_fn()
        print('target network keys:', self.target_network.state_dict().keys())
        self.target_network.load_state_dict(self.network.state_dict(), strict=False)
        print('loaded.')
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None
        self.data_params = config.data_params
        self.debug_opts = config.ddpg_debug
        self._meter = AverageMeter()
        self._timer = MultiTimer()
        self.noise_sigma = config.noise_sigma
        # Input data normalization for eps schedule and attack (not for training)
        if self.data_params["method"] == 'min_max':
            self.logger.info("using data min and max for data normalization")
            self.state_min, self.state_max = zip(*self.data_params['state_space_range'])
            self.state_min = torch.tensor(self.state_min, dtype=torch.float32).to(Config.DEVICE)
            self.state_max = torch.tensor(self.state_max, dtype=torch.float32).to(Config.DEVICE)
        elif self.data_params["method"] == 'mean_std':
            self.logger.info("using data mean and stddev for data normalization")
            self.state_mean = torch.tensor(self.data_params['state_mean'], dtype=torch.float32).to(Config.DEVICE)
            self.state_std = torch.tensor(self.data_params['state_std'], dtype=torch.float32).to(Config.DEVICE)
            self.action_std = torch.tensor(self.data_params['action_std'], dtype=torch.float32).to(Config.DEVICE)
            # avoid division by 0
            self.state_std += 1e-10
            self.action_std += 1e-10
        # Attack related parameters
        attack_config = self.config.attack_params
        self.enabled_attack = attack_config['enabled']

        if attack_config['enabled']:
            if self.data_params["method"] == 'min_max':
                self.state_min = self.state_min.view(1, -1)
                self.state_max = self.state_max.view(1, -1)
            elif self.data_params["method"] == 'mean_std':
                self.state_mean = self.state_mean.view(1, -1)
                self.state_std = self.state_std.view(1, -1)
                self.action_std =  self.action_std.view(1, -1)
            else:
                raise ValueError("normalization method must be specified for attack")
            self.attack_type = attack_config['type']
            self.attack_epsilon = attack_config['eps']
            self.attack_iteration = attack_config['iteration']
            self.attack_alpha = attack_config['alpha']
        # Robust training related parameters
        robust_config = self.config.robust_params
        if robust_config['enabled']:
            self.advtrain_scheduler = LinearSchedule(**robust_config['advtrain_scheduler'])
            self.robust_eps_scheduler = LinearSchedule(**robust_config['eps_scheduler'])
            self.robust_beta_scheduler = LinearSchedule(**robust_config['beta_scheduler'])
            # A rough range of each state variable, such that we can use correct eps for each dimension
            if self.data_params["method"] == 'min_max':
                self.state_range = self.state_max - self.state_min
            elif self.data_params["method"] == 'mean_std':
                self.state_range = self.state_std
                self.action_range = self.action_std
            else:
                raise ValueError("robust training requires a data range to determine eps")
        self.logger.info('Actor network: %s', self.network.fc_action)
        self.logger.info('Critic network: %s', self.network.fc_critic)
            
        # load sarsa 
        if attack_config['enabled'] and attack_config['type'].startswith('sarsa'):
            sarsa_params = config.sarsa_params
            self.sarsa_params = sarsa_params
            self.suffix =  "{start}_{end}_{steps}_{start_step}".format(**sarsa_params['action_eps_scheduler'] ) + "_{}".format(sarsa_params['sarsa_reg'])
            self.sarsa_action_ratio = attack_config['sarsa_action_ratio']

    # We need to save the target network's weights as well
    def save(self, filename):
        super(RobustDDPGAgent, self).save(filename)
        torch.save(self.target_network.state_dict(), '%s.target_model' % (filename))
        # Save the replay buffer for the best model
        if 'best' in os.path.basename(filename):
            self.replay.dump('%s.rb' % (filename))
        # Save optimizer states
        torch.save(self.network.actor_opt.state_dict(), '%s.actor_opt' % (filename))
        torch.save(self.network.critic_opt.state_dict(), '%s.critic_opt' % (filename))

    # We need to update the target network's weights as well
    def load(self, filename):
        super(RobustDDPGAgent, self).load(filename)
        if os.path.exists('%s.target_model' % filename):
            self.logger.info("Found target model. Loading it instead of copying model.")
            state_dict = torch.load('%s.target_model' % filename, map_location=lambda storage, loc: storage)
            # Load target network.
            self.target_network.load_state_dict(state_dict, strict=True)
        else:
            self.logger.info("Did not find target model. Using the main model's parameters.")
            self.target_network.load_state_dict(self.network.state_dict())
        if self.replay is not None and os.path.exists('%s.rb' % filename):
            self.logger.info("Found replay buffer. Loading replay buffer!")
            self.replay.clear()
            self.replay.load('%s.rb' % filename)
            self.logger.info("Replay buffer has %s elements.", self.replay.size())
        # Load optimzier states
        if self.network.actor_opt is not None and os.path.exists('%s.actor_opt' % filename):
            self.logger.info("Found actor optimizer states!")
            actor_opt_state_dict = torch.load('%s.actor_opt' % filename, map_location=lambda storage, loc: storage)
            self.network.actor_opt.load_state_dict(actor_opt_state_dict)
        if self.network.critic_opt is not None and os.path.exists('%s.critic_opt' % filename):
            self.logger.info("Found critic optimizer states!")
            critic_opt_state_dict = torch.load('%s.critic_opt' % filename, map_location=lambda storage, loc: storage)
            self.network.critic_opt.load_state_dict(critic_opt_state_dict)


    def load_sarsa(self, filename):
        self.logger.info('Load Sarsa network: %s.model_sarsa_%s' %(filename, self.suffix),)
        state_dict = torch.load('%s.model_sarsa_%s' %(filename, self.suffix), map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        normalizer_file = '%s.stats_sarsa_%s' % (filename,self.suffix)
        if os.path.exists(normalizer_file):
            with open(normalizer_file, 'rb') as f:
                self.config.state_normalizer.load_state_dict(pickle.load(f))
        else:
            self.logger.info("Not intializing normalizer because {} does not exist.".format(normalizer_file))

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_((target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix).detach_())

    def eval_step(self, state, certify_eps=0.0):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        if self.enabled_attack:
            if self.attack_type == "action":
                state = self.attack_action(state)
            elif self.attack_type == "random":
                state = self.attack_random(state)
            elif self.attack_type == 'critic':
                state = self.attack_critic(state)
            elif self.attack_type == 'sarsa':
                state = self.attack_critic(state)
            elif self.attack_type == 'sarsa_action':
                state = self.attack_critic_action(state)
            else:
                raise NotImplementedError
        # if self.noise_sigma != 0:
        #     state += to_np( tensor(self.noise_sigma * np.random.randn(*state.shape) )  * self.state_std)
        action = self.network(state)
        z = torch.zeros(size=(1, state.shape[1]))
        if certify_eps > 0.0:
            state = torch.from_numpy(state.astype(np.float32)).to(Config.DEVICE)
            scaled_robust_eps = self.state_range * certify_eps
            if False:
                # During evaluation always use no loss fusion to compute bounds.
                actor_l2 = self.network.actor_bound(phi_lb=state - scaled_robust_eps, phi_ub=state + scaled_robust_eps, beta=0.0, phi = state, center = action)
                actor_l2 = actor_l2.item()
                actor_l1 = 0
                actor_linf = 0
                actor_diff = 0
            else:
                actor_ub, actor_lb = self.network.actor_bound(phi_lb=state - scaled_robust_eps, phi_ub=state + scaled_robust_eps, phi = state, beta=0.0, upper=True, lower=True)
                actor_ub.tanh_()
                actor_lb.tanh_()
                # batch size is 1 for evaluation
                actor_diff = torch.max(actor_ub - action, action - actor_lb)[:1]
                actor_linf = torch.norm(actor_diff, p=float('inf'), dim=1).detach().mean().item()
                actor_l2 = torch.norm(actor_diff, p=2.0, dim=1).detach().mean().item()
                actor_l1 = torch.norm(actor_diff, p=1.0, dim=1).detach().mean().item()
                actor_diff = actor_diff.mean().item()
            self.config.state_normalizer.unset_read_only()
            return to_np(action), actor_l1, actor_l2, actor_linf, actor_diff
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def attack_random(self, state):
        self.config.state_normalizer.set_read_only()
        dtype = state.dtype
        ori_state = state.copy()
        state = self.config.state_normalizer(state)
        state = self.normalize( tensor(state) ) #convert to tensor
        noise = np.random.uniform(-self.attack_epsilon , self.attack_epsilon , state.data.shape).astype(dtype)
        state = tensor( noise ) + state
        state = self.denormalize(state)
        return to_np(state)

    def attack_action(self, state):
        self.config.state_normalizer.set_read_only()
        dtype = state.dtype
        state = self.config.state_normalizer(state)
        state = tensor(state)
        gt_action = self.network.actor(state).clone().detach() 
        gt_action = self.action_normalize(gt_action)


        criterion = torch.nn.MSELoss()
        ori_state = self.normalize( state.clone().detach() )
        # self.attack_epsilon = 0.1
        # random start ("alpha" is the per-step perturbation size)
        noise = np.random.uniform(-self.attack_alpha , self.attack_alpha , state.data.shape).astype(dtype)

        state = tensor(noise) + ori_state # normalized 
        state = self.denormalize(state)

        for _ in range(self.attack_iteration):
            # state.requires_grad = True

            state = self.network.feature(state.clone().detach()).requires_grad_(True)
            action = self.network.actor(state)
            action = self.action_normalize(action)

            loss = -criterion(action, gt_action)
            self.network.fc_action.zero_grad()
            loss.backward()
            adv_state = self.normalize(state) - self.attack_alpha * state.grad.sign()
            state = self.denormalize( torch.min( torch.max(adv_state , ori_state-self.attack_epsilon), ori_state+self.attack_epsilon) )
        return to_np(state)

    def attack_critic(self, state, attack_epsilon = None, attack_iteration = None, attack_stepsize = None):
        # Backward compatibility, use values read in config file
        attack_epsilon = self.attack_epsilon if attack_epsilon is None else attack_epsilon
        attack_stepsize = self.attack_alpha if attack_stepsize is None else attack_stepsize
        attack_iteration = self.attack_iteration if attack_iteration is None else attack_iteration
        dtype = state.dtype
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        state = self.network.feature(state) #convert to tensor
        # ori_state = self.normalize(state.data)
        ori_state_tensor = tensor(state.clone().detach())
        ori_state = self.normalize( state.clone().detach())
        # random start
        noise = np.random.uniform(-attack_stepsize, attack_stepsize, state.data.shape).astype(dtype)

        state = tensor(noise) + ori_state # normalized 
        state = self.denormalize(state)

        # self.attack_epsilon = 0.1
        state_ub = ori_state + attack_epsilon
        state_lb = ori_state - attack_epsilon
        for _ in range(attack_iteration):
            state = self.network.feature(state.clone().detach()).requires_grad_(True)
            action = self.network.actor(state)
            qval = self.network.critic(ori_state_tensor, action)
            loss = torch.mean(qval)
            loss.backward()
            adv_state = self.normalize(state) - attack_stepsize * state.grad.sign()
            # adv_state = self.normalize(state) + 0.01 * state.grad.sign()
            state = self.denormalize(torch.min(torch.max(adv_state , state_lb), state_ub))
            # state =  torch.max(torch.min(adv_state, self.state_max), self.state_min)
        self.network.fc_critic.zero_grad()
        self.network.fc_action.zero_grad()
        
        return to_np(state) 

    def attack_critic_action(self, state):
        self.config.state_normalizer.set_read_only()
        dtype = state.dtype
        state = self.config.state_normalizer(state)
        state = tensor(state)
        ori_state_tensor = tensor(state.clone().detach())
        gt_action = self.network.actor(state).clone().detach() 
        gt_action = self.action_normalize(gt_action)

        criterion = torch.nn.MSELoss()
        ori_state = self.normalize( state.clone().detach() )
        # random start 
        noise = np.random.uniform(-self.attack_alpha , self.attack_alpha , state.data.shape).astype(dtype)

        state = tensor(noise) + ori_state # normalized 
        state = self.denormalize(state)
        for _ in range(self.attack_iteration):
            # state.requires_grad = True

            state = self.network.feature(state.clone().detach()).requires_grad_(True)
            action = self.network.actor(state)
            # mse loss
            qval = self.network.critic(ori_state_tensor, action)
            loss1 = torch.mean(qval)
            
            loss2 = -self.sarsa_action_ratio * criterion(self.action_normalize(action), gt_action)
            if self.sarsa_action_ratio != 1:
                loss = (1-self.sarsa_action_ratio) * loss1 + loss2
            else:

                loss = loss1 + loss2

            self.network.fc_action.zero_grad()
            self.network.fc_critic.zero_grad()
            loss.backward()
            adv_state = self.normalize(state) - self.attack_alpha * state.grad.sign()
            state = self.denormalize( torch.min( torch.max(adv_state , ori_state-self.attack_epsilon), ori_state+self.attack_epsilon) )
        return to_np(state)
   
    # Normalization are currently only used for attack.
    def normalize(self, state):
        if self.data_params["method"] == "min_max":
            state = (state - self.state_min) /  ( self.state_max - self.state_min)
        elif self.data_params["method"] == "mean_std":
            state = (state - self.state_mean) / self.state_std
        elif self.data_params["method"] == "none":
            return state
        else:
            raise ValueError("unknown normalization method")
        return state
    def denormalize(self,state):
        if self.data_params["method"] == "min_max":
            state = state * (self.state_max - self.state_min) + self.state_min
        elif self.data_params["method"] == "mean_std":
            state = state * self.state_std + self.state_mean
        elif self.data_params["method"] == "none":
            return state
        else:
            raise ValueError("unknown normalization method")
        return state
    def action_normalize(self, action):
        action = action / self.action_std
        return action 
    def action_denormalize(self, action): 
        action = action * self.action_std
        return action 


    def step(self):
        config = self.config
        robust_config = self.config.robust_params
        if robust_config['enabled']:
            advtrain_eps = self.advtrain_scheduler()
            robust_eps = self.robust_eps_scheduler()
            robust_beta = self.robust_beta_scheduler()
            # rescale eps based on each element's range
            scaled_robust_eps = self.state_range * robust_eps
            strategy_opts = robust_config['strategy_opts']
            actor_lb = actor_ub = None
        else:
            advtrain_eps = robust_eps = robust_beta = 0.0
        
        self._timer.start('total')
        self._timer.start('action')
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state).astype(np.float32)

        if self.total_steps < config.warm_up and not self.config.load_pretrain:
            # when a pretrained model is loaded, do not use random sample
            action = [self.task.action_space.sample()]
        else:
            if robust_config['enabled'] and advtrain_eps > 1e-10 and 'adv_training' in robust_config['strategy']:
                # Use adversarial training for the agent
                if strategy_opts['adv_ratio'] >= random.random():
                    # Only attack a portion of frames
                    action_state = self.attack_critic(
                            self.state, attack_epsilon=advtrain_eps, attack_iteration=strategy_opts['pgd_steps'], attack_stepsize=advtrain_eps/strategy_opts['pgd_steps'])
                else:
                    action_state = self.state
            else:
                action_state = self.state
            with torch.no_grad():
                action = self.network(action_state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        self._timer.stop('action')

        self._timer.start('env')
        if config.show_game:
            for env in self.task.env.envs:
                # Render Mujuco animation
                env.unwrapped.render()
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state).astype(np.float32)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)
        self._meter.update('reward', float(reward))
        self._timer.stop('env')


        self._timer.start('replay_buf')
        experiences = list(zip(self.state, action, reward, next_state, done))
        self.replay.feed_batch(experiences)
        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1
        self._timer.stop('replay_buf')

        if self.replay.size() >= config.warm_up:
            self._timer.start('replay_buf')
            experiences = self.replay.sample()
            self._timer.stop('replay_buf')

            self._timer.start('data')
            states, actions, rewards, next_states, terminals = experiences
            states = tensor(states)
            actions = tensor(actions)
            rewards = tensor(rewards).unsqueeze(-1)
            next_states = tensor(next_states)
            mask = tensor(1 - terminals).unsqueeze(-1)
            self._timer.stop('data')

            self._timer.start('q_net')
            # Regular training for Q learning
            with torch.no_grad():
                phi_next = self.target_network.feature(next_states)
                a_next = self.target_network.actor(phi_next)
                if robust_config['enabled'] and robust_eps > 1e-10 and 'critic_minimax' in robust_config['strategy']:
                    # minimax loss for the predicted Q value (from target network)
                    # this is always the worst possible Q considering the state perturbation. We are learning a Q function that represents best worst state perturbation values.
                    _, q_next = self.target_network.critic_bound(
                            phi_lb=phi_next - scaled_robust_eps, phi_ub=phi_next + scaled_robust_eps, a_lb=a_next.detach(), a_ub=a_next.detach(), beta=robust_beta, upper=False, lower=True, phi=phi_next, action=a_next)
                else:
                    q_next = self.target_network.critic(phi_next, a_next)
                self._meter.update('q_next', q_next.mean().item())
                q_next = config.discount * mask * q_next
                q_next.add_(rewards)
                q_next = q_next.detach()
            phi = self.network.feature(states)
            q = self.network.critic(phi, actions)
            self._meter.update('q', q.mean().item())
            # Q-learning loss
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()
            self._timer.stop('q_net')
            
            self._timer.start('p_net')
            phi = self.network.feature(states)
            action = self.network.actor(phi)
            self._timer.stop('p_net')

            self._timer.start('critic_reg')
            if robust_config['enabled'] and robust_eps > 1e-10:
                if 'critic_reg' in robust_config['strategy']:
                    # with actor bounds
                    # actor network bound (input: state, output: action range)
                    actor_ub, actor_lb = self.network.actor_bound(phi_lb=phi - scaled_robust_eps, phi_ub=phi + scaled_robust_eps, phi=phi, beta=robust_beta, upper=True, lower=True)
                    actor_ub = torch.tanh(actor_ub)
                    actor_lb = torch.tanh(actor_lb)
                    self._meter.update('act_lb', actor_lb.mean().item())
                    self._meter.update('act_ub', actor_ub.mean().item())
                    # Regularize Q function
                    critic_ub, critic_lb = self.network.critic_bound(
                            phi_lb=phi - scaled_robust_eps, phi_ub=phi + scaled_robust_eps, a_lb=actor_lb.detach(), a_ub=actor_ub.detach(), beta=robust_beta, upper=True, lower=True, phi=phi, action=action)
                    self._meter.update('cri_lb', critic_lb.mean().item())
                    self._meter.update('cri_ub', critic_ub.mean().item())
                    critic_reg_loss = (critic_ub - critic_lb).mean()
                    self._meter.update('cri_reg_loss', critic_reg_loss.item())
                    self._meter.update('cri_loss_no_reg', critic_loss.item())
                    critic_loss += strategy_opts['critic_reg'] * critic_reg_loss
                if 'critic_reg_no_act' in robust_config['strategy']:
                    # without actor bounds
                    critic_ub, critic_lb = self.network.critic_bound(
                            phi_lb=phi - scaled_robust_eps, phi_ub=phi + scaled_robust_eps, a_lb=action, a_ub=action, beta=robust_beta, upper=True, lower=True, phi=phi, action=action)
                    self._meter.update('cri_lb', critic_lb.mean().item())
                    self._meter.update('cri_ub', critic_ub.mean().item())
                    critic_reg_loss = (critic_ub - critic_lb).mean()
                    self._meter.update('cri_reg_loss', critic_reg_loss.item())
                    self._meter.update('cri_loss_no_reg', critic_loss.item())
                    critic_loss += strategy_opts['critic_reg'] * critic_reg_loss
            self._timer.stop('critic_reg')

            self._timer.start('q_net')
            self.network.fc_critic.zero_grad()
            if robust_config['enabled'] and 'critic_reg_no_act' in robust_config['strategy']:
                critic_loss.backward(retain_graph=True)
            else:
                critic_loss.backward(retain_graph=False)
            self.network.critic_opt.step()
            self._timer.stop('q_net')

            self._timer.start('p_net')
            # Policy gradient loss
            if robust_config['enabled'] and robust_eps > 1e-10 and 'actor_minimax' in robust_config['strategy']:
                # Policy loss is a lower bound under state perturbation. The actor learns how to find an action that maximizes the lower bound of critic under perturbation
                if 'critic_reg_no_act' in robust_config['strategy']:
                    lb = critic_lb
                else:
                    _, lb = self.network.critic_bound(phi_lb=phi - scaled_robust_eps, phi_ub=phi + scaled_robust_eps, a_lb=action, a_ub=action, beta=robust_beta, upper=False, lower=True)
                policy_loss = -lb.mean()
            else:
                policy_loss = -self.network.critic(phi.detach(), action).mean()
            if robust_config['enabled'] and robust_eps > 1e-10:
                # Robust actor regularizations.
                if 'actor_reg' in robust_config['strategy'] or 'actor_tv_reg' in robust_config['strategy'] or 'actor_l2_reg' in robust_config['strategy']:
                    # actor network bound (input: state, output: action range)
                    if 'use_sgld' in robust_config['strategy']:
                        # Use SGLD based method to find a lower bound
                        if 'actor_l2_reg' in robust_config['strategy']:
                            steps = strategy_opts['sgld_steps']
                            step_eps = scaled_robust_eps / steps
                            # upper and lower bounds for clipping
                            adv_ub = phi + scaled_robust_eps
                            adv_lb = phi - scaled_robust_eps
                            # add uniform noise beween +/- scaled_robust_eps
                            # SGLD noise factor. We set (inverse) beta=1e-5 as gradients are relatively small here.
                            beta = 1e-5
                            noise_factor = torch.sqrt(2 * step_eps) * beta
                            noise = torch.randn_like(phi) * noise_factor
                            # First SGLD step, the gradient is 0, so only need to add noise. Project to Linf box.
                            adv_phi = (phi.clone() + noise.sign() * step_eps).detach().requires_grad_()
                            # and clip into the upper and lower bounds (not necessary for now as we use uniform noise)
                            # adv_phi = torch.max(adv_phi, adv_lb)
                            # adv_phi = torch.min(adv_phi, adv_ub)
                            for i in range(steps):
                                # Find a nearby state adv_phi that maximize the difference
                                adv_loss = (self.network.actor(adv_phi) - action.detach()).pow(2).mean()
                                # Need to clear gradients before the backward() for policy_loss
                                adv_loss.backward()
                                # Reduce noise at every step. We start at step 2.
                                noise_factor = torch.sqrt(2 * step_eps) * beta / (i+2)
                                # Project noisy gradient to step boundary.
                                update = (adv_phi.grad + noise_factor * torch.randn_like(adv_phi)).sign() * step_eps
                                adv_phi = adv_phi + update
                                # clip into the upper and lower bounds
                                adv_phi = torch.max(adv_phi, adv_lb)
                                adv_phi = torch.min(adv_phi, adv_ub).detach().requires_grad_()
                            # see how much the difference is
                            self._meter.update('sgld_act_diff', (adv_phi - phi).abs().sum().item())
                            # We want to minimize the loss
                            action_reg_loss = (self.network.actor(adv_phi) - action).pow(2).mean()
                        else:
                            raise(ValueError("unsupported SGLD loss!"))
                    else:
                        # Use convex relaxation method to find a upper bound
                        if robust_config.get('use_loss_fusion', False):
                            # Bound the L2 loss directly.
                            if 'actor_l2_reg' in robust_config['strategy']:
                                action_reg_loss = self.network.actor_bound(phi_lb=phi - scaled_robust_eps, phi_ub=phi + scaled_robust_eps, beta=robust_beta, upper=True, lower=False, phi=phi, center=action)
                                action_reg_loss = action_reg_loss.mean()
                            else:
                                raise(ValueError("unsupported actor-reg loss!"))
                        else:
                            # Bound last layer ub and lb and then use IBP to bound the loss.
                            if actor_ub is None:
                                actor_ub, actor_lb = self.network.actor_bound(phi_lb=phi - scaled_robust_eps, phi_ub=phi + scaled_robust_eps, beta=robust_beta, upper=True, lower=True, phi=phi)
                                actor_ub = torch.tanh(actor_ub)
                                actor_lb = torch.tanh(actor_lb)
                                self._meter.update('act_lb', actor_lb.mean().item())
                                self._meter.update('act_ub', actor_ub.mean().item())
                            if 'actor_reg' in robust_config['strategy']:
                                action_reg_loss = (actor_ub - actor_lb).mean()
                            elif 'actor_tv_reg' in robust_config['strategy']:
                                action_reg_loss = torch.max(actor_ub - action, action - actor_lb).mean()
                            elif 'actor_l2_reg' in robust_config['strategy']:
                                action_reg_loss = torch.max(actor_ub - action, action - actor_lb).pow(2).mean()
                    self._meter.update('act_reg_loss', action_reg_loss.item())
                    self._meter.update('act_loss_no_reg', policy_loss.item())
                    policy_loss += strategy_opts['actor_reg'] * action_reg_loss
            self.network.fc_action.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)
            self._timer.stop('p_net')

            self._meter.update('critic_loss', critic_loss.item())
            self._meter.update('policy_loss', policy_loss.item())
            self._timer.stop('total')
            if self.total_steps % self.debug_opts["print_frame"] == 0:
                if robust_config['enabled']:
                    robust_info = "rob_eps={:.5f} rob_beta={:.5f}".format(robust_eps, robust_beta)
                    if 'adv_training' in robust_config['strategy']:
                        robust_info += " advtrain_eps={:.5f}".format(advtrain_eps)
                else:
                    robust_info = ""
                self.logger.info("steps={} {} {} {}".format(
                    self.total_steps, self._meter, self._timer if self.debug_opts["profile_time"] else "", robust_info))
                # compute average over next "print_frame" steps
                self._meter.reset()
                self._timer.reset()
        else:
            self._timer.stop('total')

