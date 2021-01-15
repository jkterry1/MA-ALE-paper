import argparse
from all.environments.multiagent_atari import MaxAndSkipMAALE
from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
from all.presets import atari
from all.presets.multiagent_atari import independent
from all.core import State
from timeit import default_timer as timer
import torch
import os
import sys
import numpy as np
import random
import subprocess
import gym
import importlib
import time

from supersuit.base_aec_wrapper import BaseWrapper
from PIL import Image
from gym.wrappers import AtariPreprocessing
from pettingzoo.atari.base_atari_env import BaseAtariEnv, base_env_wrapper_fn, parallel_wrapper_fn
from pettingzoo.utils.to_parallel import from_parallel
from supersuit import resize_v0, frame_skip_v0

color_map = {104: 110, 110: 167, 179: 85, 149: 157}
#color_map = {110: 104, 167: 110, 85: 179, 157: 149}
class ObservationWrapper(BaseWrapper):
    def _modify_action(self, agent, action):
        return action

    def _update_step(self, agent):
        pass

class recolor_observations(ObservationWrapper):
    def _check_wrapper_params(self):
        pass

    def _modify_spaces(self):
        pass

    def _modify_observation(self, agent, observation):
        new_obs = np.zeros_like(observation)
        mask1 = (observation == 104)
        mask2 = (observation == 110)
        mask3 = (observation == 179)
        mask4 = (observation == 149)
        new_obs[mask1] = 90
        new_obs[mask2] = 147
        new_obs[mask3] = 64
        new_obs[mask4] = 167
        return new_obs

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class MultiAgentAtariEnv():
    def __init__(self, env_name, device='cpu', frame_skip=4):
        def raw_env(**kwargs):
            return BaseAtariEnv(game="surround", num_players=1, mode_num=2, **kwargs)
        env = parallel_wrapper_fn(base_env_wrapper_fn(raw_env))(obs_type="grayscale_image")
        env = MaxAndSkipMAALE(env, skip=frame_skip)
        env = from_parallel(env)
        env = recolor_observations(env)
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


class TestRandom:
    def __init__(self):
        pass
    def act(self, state):
        return random.randint(0,17)

def generate_episode_gifs(env, _agent, max_frames, dir):
    # initialize the episode
    observation = env.reset()['observation'].cpu().numpy()
    print(observation)
    frame_idx = 0
    prev_obs = None

    # loop until the episode is finished
    done = False
    while not done:
        action = _agent.act("first_0", State.from_gym((observation.reshape((1, 84, 84),)), device=device, dtype=np.uint8))
        if not isinstance(action, int):
            action = action.cpu().numpy()[0]
        state = env.step(action)
        reward = state['reward']
        obseration = state['observation'].cpu().numpy()
        done = state['done']
        if reward != 0.0:
            print(reward)
        obs = env._env.env.env.env.env.env.aec_env.env.env.env.ale.getScreenRGB()
        if not prev_obs or not np.equal(obs, prev_obs).all():
            im = Image.fromarray(obs)
            im.save(f"{dir}{str(frame_idx).zfill(4)}.png")

            frame_idx += 1
            if frame_idx >= max_frames:
                break

def test_single_episode(env, _agent, generate_gif_callback=None):
    # initialize the episode
    observation, _, _, _ = env.reset()
    observation = env.reset()['observation'].cpu().numpy()
    np.set_printoptions(threshold=sys.maxsize, linewidth=500)
    #print(observation)
    #obs = env._env.env.env.env.env.aec_env.env.env.env.ale.getScreenRGB()
    #print(obs[:,:,0])
    returns = 0
    num_steps = 0
    frame_idx = 0
    prev_obs = None

    # loop until the episode is finished
    done = False
    while not done:
        action = _agent.act("first_0", State.from_gym((observation.reshape((1, 84, 84),)), device=device, dtype=np.uint8))
        if not isinstance(action, int):
            action = action.cpu().numpy()[0]
        state = env.step(action)
        reward = state['reward']
        obseration = state['observation'].cpu().numpy()
        done = state['done']
        returns += reward
        num_steps += 1

    return returns, num_steps

def test_independent(env, agent, frames):
    returns = []
    num_steps = 0
    while num_steps < frames:
        episode_return, ep_steps = test_single_episode(env, agent)
        returns.append(episode_return)
        num_steps += ep_steps
        print(num_steps)
        # self._log_test_episode(episode, episode_return)
    return returns

def returns_agent(returns, agent):
    print(returns)
    print(np.mean(returns))
    return np.mean(returns)

def main():
    parser = argparse.ArgumentParser(description="Run an multiagent Atari benchmark.")
    parser.add_argument("checkpoint", help="Name of the checkpoint (e.g. pong_v1).")
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--frames", type=int, default=100000, help="The number of training frames."
    )
    parser.add_argument(
        "--agent", type=str, default="first_0", help="Agent to print out value."
    )
    parser.add_argument(
        "--generate-gif", action="store_true", help="Agent to print out value."
    )
    # parser.add_argument(
    #     "--render", type=bool, default=False, help="Render the environment."
    # )
    args = parser.parse_args()

    # self._writer = ExperimentWriter(self, multiagent.__name__, env.name, loss=write_loss)
    # self._writers = {
    #     agent : ExperimentWriter(self, "{}_{}".format(multiagent.__name__, agent), env.name, loss=write_loss)
    #     for agent in env.agents
    # }
    frame_skip = 1 if args.generate_gif else 4
    env = MultiAgentAtariEnv("surround_v1", device=args.device, frame_skip=frame_skip)

    #def raw_env():
    #    return BaseAtariEnv(game='surround', num_players=1, mode_num=2, full_action_space=True, obs_type="grayscale_image")

    #env = parallel_wrapper_fn(raw_env)()
    #env = MaxAndSkipMAALE(env, skip=frame_skip)
    #env = from_parallel(env)
    #env = resize_v0(env, 84, 84)
    env.reset()
    # base_builder = getattr(atari, agent_name)()
    preset = torch.load(os.path.join("./checkpoints/latest_ind_atari_checkpoints/" + args.checkpoint + "_final_checkpoint.th"))
    agent = preset.test_agent()

    if not args.generate_gif:
        returns = test_independent(env, agent, args.frames)
        agent_names = ["first_0", "second_0", "third_0", "fourth_0"]
        with open("./outfiles/" + args.checkpoint + ".txt",'w') as out:
            out.write(f"Environment: Surround\n")
            out.write(f"Checkpoint Name: {args.checkpoint}\n")
            out.write(f"Agent: {args.agent}\n")
            out.write(f"Returns: {str(returns)}\n")
            out.write(f"Average return: {str(np.mean(returns))}\n")
            out.write(f"Evaluation frames: {args.frames}\n")

        #print(returns_agent1(returns))
        # preset = independent({agent:base_builder for agent in env.agents}).env(env).hyperparameters(replay_buffer_size=350000,replay_start_size=100)
        # agent = preset.agent()
    else:
        name = f"Surround_{args.agent}"
        folder = f"frames/{name}/"
        os.makedirs(folder,exist_ok=True)
        os.makedirs("gifs",exist_ok=True)
        generate_episode_gifs(env, agent, args.frames, folder)

        ffmpeg_command = [
            "ffmpeg",
            "-framerate", "60",
            "-i", f"frames/{name}/%04d.png",
            "-vcodec", "libx264",
            "-crf", "1",
            "-pix_fmt", "yuv420p",
            f"gifs/{name}.mp4"
        ]
        # ffmpeg_command = [
        #     "convert",
        #     '-delay','1x120',
        #      f"frames/{name}/*.png",
        #     f"gifs/{name}.gif"
        # ]
        subprocess.run(ffmpeg_command)

if __name__ == "__main__":
    main()
