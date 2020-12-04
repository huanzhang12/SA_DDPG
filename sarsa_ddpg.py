from deep_rl.network import *
from deep_rl.component import *
from deep_rl.agent.BaseAgent import *
from deep_rl.utils.schedule import LinearSchedule
import torch
import torch.nn as nn
import torchvision
import copy

from utils import AverageMeter, MultiTimer

from auto_LiRPA.bound_ops import BoundParams

# Specialized DDPG Agent for trainig a Sarsa value function.
# The policy is not trained and kept fixed. Only train critic
class RobustDDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None
        self.data_params = config.data_params
        self.noise_sigma = config.noise_sigma
        self.debug_opts = config.ddpg_debug
        self._meter = AverageMeter()
        self._timer = MultiTimer()
        # Robust training related parameters
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

        ## add sara parameters 
        sarsa_params = copy.deepcopy(config.sarsa_params)
        self.sarsa_params = sarsa_params

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
                self.action_std = self.action_std.view(1, -1)
            else:
                raise ValueError("normalization method must be specified for attack")
            self.attack_type = attack_config['type']
            self.attack_epsilon = attack_config['eps']
            self.attack_iteration = attack_config['iteration']
            self.attack_alpha = attack_config['alpha']

        self.suffix =  "{start}_{end}_{steps}_{start_step}".format( **sarsa_params['action_eps_scheduler'] ) + "_{}".format(sarsa_params['sarsa_reg'])
        # Shift start step.
        sarsa_params['eps_scheduler']['start_step'] += self.config.warm_up
        sarsa_params['beta_scheduler']['start_step'] += self.config.warm_up
        sarsa_params['action_eps_scheduler']['start_step'] += self.config.warm_up
        self.robust_eps_scheduler = LinearSchedule(**sarsa_params['eps_scheduler'])
        self.robust_action_eps_scheduler = LinearSchedule(**sarsa_params['action_eps_scheduler'])
        self.robust_beta_scheduler = LinearSchedule(**sarsa_params['beta_scheduler'])
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
           
    def save(self, filename):
        # Save the Sarsa model for attack.
        torch.save(self.network.state_dict(), '%s.model_sarsa_%s' % (filename, self.suffix))
        with open('%s.stats_sarsa_%s' % (filename, self.suffix), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)


    def load(self, filename):
        super(RobustDDPGAgent, self).load(filename)
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, BoundParams):
                params = m.forward_value
                if params.ndim == 2:
                    torch.nn.init.kaiming_uniform_(params, a=np.sqrt(5))
                else:
                    torch.nn.init.normal_(params)
        # We need to re-initialize the critic, not using the old one.
        self.network.fc_critic.apply(weight_reset)



    def eval_step(self, state, certify_eps=0.0):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        
        if self.noise_sigma != 0:
            state += to_np( tensor(self.noise_sigma * np.random.randn(*state.shape) )  * self.state_std)
        action = self.network(state)
        if certify_eps > 0.0:
            state = torch.from_numpy(state.astype(np.float32)).to(Config.DEVICE)
            scaled_robust_eps = self.state_range * certify_eps
            actor_ub, actor_lb = self.network.actor_bound(phi_lb=state - scaled_robust_eps, phi_ub=state + scaled_robust_eps, beta=0.0, upper=True, lower=True, phi=state)
            actor_ub.tanh_()
            actor_lb.tanh_()
            # batch size is 1 for evaluation
            actor_diff = torch.max(actor_ub - action, action - actor_lb)[:1]
            actor_linf = torch.norm(actor_diff, p=float('inf'), dim=1).detach().mean().item()
            actor_l2 = torch.norm(actor_diff, p=2.0, dim=1).detach().mean().item()
            actor_l1 = torch.norm(actor_diff, p=1.0, dim=1).detach().mean().item()
            self.config.state_normalizer.unset_read_only()
            return to_np(action), actor_l1, actor_l2, actor_linf, actor_diff.mean().item()
        self.config.state_normalizer.unset_read_only()
        return to_np(action)


    def soft_update(self, target, src):
        mix_ratio = self.sarsa_params['target_network_mix']
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_((target_param * (1.0 - mix_ratio) +
                               param * mix_ratio).detach_())


    def step(self):
        config = self.config
        sarsa_config = self.sarsa_params
        robust_eps = self.robust_eps_scheduler()
        robust_action_eps_p = self.robust_action_eps_scheduler()
        robust_beta = self.robust_beta_scheduler()
        # rescale eps based on each element's range
        scaled_robust_eps = self.state_range * robust_eps
        robust_action_eps = self.action_range * robust_action_eps_p
        
        sarsa_reg = sarsa_config['sarsa_reg']
        actor_lb = actor_ub = None

        self._timer.start('total')
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        # Fill the replay buffer with all state/action/reward pairs. 
        # Since we are not updating the policy anymore, we will not generate new samples later on.
        # All samples are generated during the warmup period.
        if self.total_steps < config.warm_up:
            # Generate action.
            self._timer.start('action')
            with torch.no_grad():
                action = self.network(self.state)
            action = to_np(action)
            # Add noise (Sarsa requires exploration)
            action += self.random_process.sample()
            action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
            self._timer.stop('action')

            # Run environment.
            self._timer.start('env')
            if config.show_game:
                for env in self.task.env.envs:
                    # Render Mujuco animation
                    env.unwrapped.render()
            next_state, reward, done, info = self.task.step(action)
            if done[0]:
                self.random_process.reset_states()
            next_state = self.config.state_normalizer(next_state)
            self.record_online_return(info)
            reward = self.config.reward_normalizer(reward)
            self._timer.stop('env')

            # Add experience to replay buffer.
            self._timer.start('replay_buf')
            experiences = list(zip(self.state, action, reward, next_state, done))
            self.replay.feed_batch(experiences)
            self.state = next_state
            self._timer.stop('replay_buf')

        self.total_steps += 1
        # Replay buffer has been filled, and we start to train value function now using Sarsa update rule.
        if self.replay.size() >= config.warm_up:
            # We purely sample and do not generate any new examples any more.
            self._timer.start('replay_buf')
            experiences = self.replay.sample()
            self._timer.stop('replay_buf')

            # Convert data to tensor.
            self._timer.start('data')
            states, actions, rewards, next_states, terminals = experiences
            states = tensor(states)
            actions = tensor(actions)
            rewards = tensor(rewards).unsqueeze(-1)
            next_states = tensor(next_states)
            mask = tensor(1 - terminals).unsqueeze(-1)
            self._timer.stop('data')

            self._timer.start('q_net')
            # Sarsa update rule.
            with torch.no_grad():
                # both regular network's and target network's policies are not updated. We only update the critic.
                phi_next = self.target_network.feature(next_states)
                a_next = self.target_network.actor(phi_next) # Actor is fixed. Actually this can be precomputed.
                q_next = self.target_network.critic(phi_next, a_next)
                self._meter.update('q_next', q_next.mean().item())
                q_next = config.discount * mask * q_next
                q_next.add_(rewards)
            phi = self.network.feature(states)
            q = self.network.critic(phi, actions)
            self._meter.update('q', q.mean().item())

            # TD loss.
            # criterion = torch.nn.SmoothL1Loss()
            # critic_loss = criterion(q, q_next)
            critic_loss = (q - q_next).pow(2).mul(0.5).mean()
            # print(critic_loss.max().item())
            self._timer.stop('q_net')

            
            # Compute the robust regularizer.
            self._timer.start('critic_reg')
            if sarsa_reg > 1e-5 and (robust_eps > 0 or robust_action_eps_p > 0):
                critic_ub, critic_lb = self.network.critic_bound(
                        phi_lb=phi - scaled_robust_eps, phi_ub=phi + scaled_robust_eps, a_lb=actions - robust_action_eps, a_ub=actions + robust_action_eps, beta=robust_beta, upper=True, lower=True, phi=phi, action=actions)
                self._meter.update('cri_lb', critic_lb.mean().item())
                self._meter.update('cri_ub', critic_ub.mean().item())
                critic_reg_loss = (critic_ub - critic_lb).mean()
                self._meter.update('cri_reg_loss', critic_reg_loss.item())
                self._meter.update('cri_loss_no_reg', critic_loss.item())
                critic_loss += sarsa_reg * critic_reg_loss
            self._timer.stop('critic_reg')


            # Run optimizer.
            self._timer.start('q_net')
            self.network.fc_critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.network.critic_opt.step()
            self._timer.stop('q_net')

            # Update target network.
            self.soft_update(self.target_network, self.network)

            # Collect statistics.
            self._meter.update('critic_loss', critic_loss.item())
            self._timer.stop('total')
            if self.total_steps % self.debug_opts["print_frame"] == 0:
                robust_info = "rob_eps={:.5f} rob_beta={:.5f} act_eps={:.5f}".format(robust_eps, robust_beta, robust_action_eps_p)
                # else:
                #     robust_info = ""
                self.logger.info("steps={} {} {} {}".format(
                    self.total_steps, self._meter, self._timer if self.debug_opts["profile_time"] else "", robust_info))
                # compute average over next "print_frame" steps
                self._meter.reset()
                self._timer.reset()
        else:
            self._timer.stop('total')
