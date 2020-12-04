#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import os
import torch
import numpy as np
from ..utils import *
import torch.multiprocessing as mp
from collections import deque, OrderedDict
from skimage.io import imsave


class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level, models_path=config.models_path)
        self.task_ind = 0

    def close(self):
        close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        normalizer_file = '%s.stats' % (filename)
        if os.path.exists(normalizer_file):
            with open(normalizer_file, 'rb') as f:
                self.config.state_normalizer.load_state_dict(pickle.load(f))
        else:
            self.logger.info("Not intializing normalizer because {} does not exist.".format(normalizer_file))

    def eval_step(self, state):
        raise NotImplementedError

    def eval_episode(self, show=None, return_states=False, certify_eps=0.0, episode_number=0, frame_skip=6):
        if show is None:
            show = self.config.show_game
        env = self.config.eval_env
        state = env.reset()
        states = []
        actions = []
        certify_losses_l1 = []
        certify_losses_l2= []
        certify_losses_linf = []
        certify_losses_range = []
        steps = 0
        frame_steps = 0
        save_dir = os.path.join(self.config.models_path, "frames", "{:03d}".format(episode_number))
        if self.config.save_frame:
            os.makedirs(save_dir, exist_ok=True)
        while True:
            if certify_eps > 0.0:
                action, certify_loss_l1, certify_loss_l2, certify_loss_linf, certify_loss_range = self.eval_step(state, certify_eps=certify_eps)
                certify_losses_l1.append(certify_loss_l1)
                certify_losses_l2.append(certify_loss_l2)
                certify_losses_linf.append(certify_loss_linf)
                certify_losses_range.append(certify_loss_range)
            else:
                action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            states.append(state[0])
            actions.append(action[0])
            for e in env.env.envs:
                # Render Mujuco animation
                if self.config.save_frame:
                    if steps % frame_skip == 0:
                        frame_steps += 1
                        image = e.unwrapped.render(mode='rgb_array')
                        imsave('%s/%04d.bmp' % (save_dir, frame_steps), image, check_contrast=False)
                if show:
                    e.unwrapped.render()
            ret = info[0]['episodic_return']
            if ret is not None:
                break
            steps += 1
        if return_states:
            return ret, states, actions, certify_losses_l1, certify_losses_l2, certify_losses_linf, certify_losses_range
        return ret

    def eval_episodes(self):
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
            self.total_steps, np.mean(episodic_returns), np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        ))
        self.logger.add_scalar('episodic_return_test', np.mean(episodic_returns), self.total_steps)
        return {
            'episodic_return_test': np.mean(episodic_returns),
        }

    def record_online_return(self, info, offset=0):
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, ret))
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def switch_task(self):
        config = self.config
        if not config.tasks:
            return
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)

    def record_episode(self, dir, env):
        mkdir(dir)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, dir, steps)
            action = self.record_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            steps += 1
            if ret is not None:
                break

    def record_step(self, state):
        raise NotImplementedError

    # For DMControl
    def record_obs(self, env, dir, steps):
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave('%s/%04d.png' % (dir, steps), obs)


class BaseActor(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config):
        mp.Process.__init__(self)
        self.config = config
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self._state = None
        self._task = None
        self._network = None
        self._total_steps = 0
        self.__cache_len = 2

        if not config.async_actor:
            self.start = lambda: None
            self.step = self._sample
            self.close = lambda: None
            self._set_up()
            self._task = config.task_fn()

    def _sample(self):
        transitions = []
        for _ in range(self.config.sgd_update_frequency):
            transitions.append(self._transition())
        return transitions

    def run(self):
        self._set_up()
        config = self.config
        self._task = config.task_fn()

        cache = deque([], maxlen=2)
        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.STEP:
                if not len(cache):
                    cache.append(self._sample())
                    cache.append(self._sample())
                self.__worker_pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            elif op == self.NETWORK:
                self._network = data
            else:
                raise NotImplementedError

    def _transition(self):
        raise NotImplementedError

    def _set_up(self):
        pass

    def step(self):
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        if not self.config.async_actor:
            self._network = net
        else:
            self.__pipe.send([self.NETWORK, net])
