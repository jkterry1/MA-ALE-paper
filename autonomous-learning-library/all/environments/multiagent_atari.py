import importlib
import numpy as np
import torch
from pettingzoo import atari
from supersuit import resize_v0, frame_skip_v0, frame_stack_v0, sticky_actions_v0, color_reduction_v0
from all.core import State
from pettingzoo.utils.wrappers import BaseWrapper as PettingzooWrap
from supersuit.parallel_wrappers import ParallelWraper
from pettingzoo.utils.to_parallel import from_parallel

class MaxAndSkipMAALE(ParallelWraper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        assert skip > 0
        self.skip = skip

    def step(self, actions):
        obs, rews, orig_dones, orig_infos = super().step(actions)
        next_agents = self.env.agents[:]
        rews = dict(**rews)
        orig_dones = dict(**orig_dones)
        dones = orig_dones
        orig_infos = dict(**orig_infos)
        # if(any(rews.values())):
        #     print(rews)
        for i in range(self.skip - 1):
            if all(dones.values()):
                break

            obs1, rews1, dones, infos = super().step(actions)
            # if(any(rews1.values())):
            #     print(rews1)
            for agent in rews1:
                obs[agent] = np.maximum(obs[agent], obs1[agent])
                rews[agent] += rews1[agent]
                orig_dones[agent] = dones[agent]
                orig_infos[agent] = infos[agent]
        # if(any(rews.values())):
        #     print("fin",rews)
        self.agents = next_agents
        return obs, rews, orig_dones, orig_infos


class MultiAgentAtariEnv():
    def __init__(self, env_name, device='cpu', frame_skip=4):
        env = importlib.import_module('pettingzoo.atari.{}'.format(env_name)).parallel_env(obs_type='grayscale_image')
        env = MaxAndSkipMAALE(env, skip=frame_skip)
        env = from_parallel(env)
        env = resize_v0(env, 84, 84)
        self._env = env
        self.name = env_name
        self.device = torch.device(device)
        self.agents = self._env.possible_agents
        self.subenvs = {
            agent : SubEnv(agent, device, self.state_spaces[agent], self.action_spaces[agent])
            for agent in self.agents
        }

    def reset(self):
        self._env.reset()
        observation, _, _, _ = self._env.last()
        state = State.from_gym((observation.reshape((1, 84, 84),)), device=self.device, dtype=np.uint8)
        return state

    def step(self, action):
        observation, reward, done, info = self._env.last()
        if self._env.dones[self._env.agent_selection]:
            action = None
        if torch.is_tensor(action):
            self._env.step(action.item())
        else:
            self._env.step(action)
        observation, reward, done, info = self._env.last()
        return State.from_gym((observation.reshape((1, 84, 84)), reward, done, info), device=self.device, dtype=np.uint8)

    def agent_iter(self):
        return self._env.agent_iter()

    @property
    def state_spaces(self):
        return self._env.observation_spaces

    @property
    def observation_spaces(self):
        return self._env.observation_spaces

    @property
    def action_spaces(self):
        return self._env.action_spaces

class SubEnv():
    def __init__(self, name, device, state_space, action_space):
        self.name = name
        self.device = device
        self.state_space = state_space
        self.action_space = action_space

    @property
    def observation_space(self):
        return self.state_space
