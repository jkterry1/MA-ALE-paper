import argparse
from all.environments import MultiAgentAtariEnv
from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
from all.presets import atari
from all.presets.multiagent_atari import independent
from all.core import State
from timeit import default_timer as timer
import torch
import os
import numpy as np
import random
import subprocess
import gym
from gym.wrappers import AtariPreprocessing

from PIL import Image


class TestRandom:
    def __init__(self):
        pass
    def act(self, state):
        return random.randint(0,17)


def generate_episode_gifs(env, _agent, max_frames, dir):
    # initialize the episode
    state = env.reset()
    frame_idx = 0
    prev_obs = None

    # loop until the episode is finished
    for agent in env.agent_iter():
        action = _agent.act(agent, state)
        state = env.step(action)
        obs = env._env.env.env.env.env.aec_env.env.env.env.env.ale.getScreenRGB()
        if not prev_obs or not np.equal(obs, prev_obs).all():
            im = Image.fromarray(obs)
            im.save(f"{dir}{str(frame_idx).zfill(4)}.png")

            frame_idx += 1
            if frame_idx >= max_frames:
                break

        if len(env._env.agents) == 1 and env._env.dones[env._env.agent_selection]:
            print(returns)
            break

def test_single_episode(env, _agent, generate_gif_callback=None):
    # initialize the episode
    observation = env.reset()
    returns = 0
    num_steps = 0
    frame_idx = 0
    prev_obs = None

    # loop until the episode is finished
    done = False
    while not done:
        print(observation.shape)
        #print(_agent.agents)
        action = _agent.act("first_0", State.from_gym((observation.reshape((1, 84, 84),)), device="cpu", dtype=np.uint8))
        print(action)
        observation, reward, done, info = env.step(action)
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
        # self._log_test_episode(episode, episode_return)
    return returns

def returns_agent(returns, agent):
    if agent not in returns[0]:
        return np.float("nan")
    agent_1_returns = [ret[agent] for ret in returns]
    return np.mean(agent_1_returns)

def main():
    parser = argparse.ArgumentParser(description="Run an multiagent Atari benchmark.")
    parser.add_argument("env", help="Name of the Atari game (e.g. Pong).")
    parser.add_argument(
        "checkpoint", help="Checkpoint number."
    )
    parser.add_argument(
        "folder", help="Folder with checkpoitns."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--vs-random", action="store_true", help="Play first_0 vs random for all other players."
    )
    parser.add_argument(
        "--agent-random", action="store_true", help="Play first_0 vs random for all other players."
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
    checkpoint_path = os.path.join(args.folder)
    print(checkpoint_path)
    frame_skip = 1 if args.generate_gif else 4
    #env = MultiAgentAtariEnv(args.env, device=args.device, frame_skip=frame_skip)
    env = gym.make('Pong-v0', full_action_space=True)
    env = AtariPreprocessing(env, frame_skip=1)
    env.reset()
    # base_builder = getattr(atari, agent_name)()
    preset = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    agent = preset.test_agent()
    if args.vs_random:
        for a in env._env.possible_agents:
            if a != args.agent:
                agent.agents[a] = TestRandom()
    if args.agent_random:
        agent.agents[args.agent] = TestRandom()

    if not args.generate_gif:
        returns = test_independent(env, agent, args.frames)
        print(returns)
        agent_names = ["first_0", "second_0", "third_0", "fourth_0"]
        open("out.txt",'w').write(f"{args.env},{args.checkpoint},{args.agent},{','.join(str(returns_agent(returns, agent)) for agent in agent_names)},{args.vs_random},{args.agent_random}\n")
        #print(returns_agent1(returns))
        # preset = independent({agent:base_builder for agent in env.agents}).env(env).hyperparameters(replay_buffer_size=350000,replay_start_size=100)
        # agent = preset.agent()
    else:
        name = f"{args.env}_{args.checkpoint}_{args.agent}_{args.vs_random}_{args.agent_random}"
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
