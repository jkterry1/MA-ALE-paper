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
import supersuit
from supersuit import color_reduction_v0, frame_stack_v1, max_observation_v0, frame_skip_v0, resize_v0
from gym.wrappers import AtariPreprocessing
import gym
from all.environments.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    FireResetEnv,
    WarpFrame,
    LifeLostEnv,
)

from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class TestRandom:
    def __init__(self):
        pass
    def act(self, state):
        return random.randint(0,17)


def generate_episode_gifs(env, _agent, max_frames, save_dir, side="first_0"):
    # initialize the episode
    observation = env.reset()
    frame_idx = 0
    prev_obs = None

    # loop until the episode is finished
    done = False
    while not done:
        #print(_agent.agents)
        action = _agent.act(side, State.from_gym((observation.reshape((1, 84, 84),)), device=device, dtype=np.uint8))
        observation, reward, done, info = env.step(action)
        if reward != 0.0:
            print(reward)
        obs = env.render(mode='rgb_array')
        if not prev_obs or not np.equal(obs, prev_obs).all():
            im = Image.fromarray(obs)
            im.save(f"{save_dir}{str(frame_idx).zfill(4)}.png")

            frame_idx += 1
            if frame_idx >= max_frames:
                break

def test_single_episode(env, _agent, generate_gif_callback=None, side="first_0"):
    # initialize the episode
    observation = env.reset()
    returns = 0
    num_steps = 0
    frame_idx = 0
    prev_obs = None
    print(side)

    # loop until the episode is finished
    done = False
    while not done:
        #print(_agent.agents)
        action = _agent.act(side, State.from_gym((observation.reshape((1, 84, 84),)), device=device, dtype=np.uint8))
        observation, reward, done, info = env.step(action)
        returns += reward
        num_steps += 1

    return returns, num_steps

def test_independent(env, agent, frames, side="first_0"):
    returns = []
    num_steps = 0
    while num_steps < frames:
        episode_return, ep_steps = test_single_episode(env, agent, side=side)
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
    parser.add_argument("env", help="Name of the Atari game (e.g. Pong).")
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
    #env = MultiAgentAtariEnv(args.env, device=args.device, frame_skip=frame_skip)
    env = gym.make(args.env, full_action_space=True)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = LifeLostEnv(env)

    
    # allow agent to see everything on the screen despite Atari's flickering screen problem
    #env = supersuit.frame_stack_v1(env, 4)
    env.reset()

    # base_builder = getattr(atari, agent_name)()
    preset = torch.load(os.path.join("./checkpoints/latest_ind_atari_checkpoints/" + args.checkpoint + "_final_checkpoint.th"))
    agent = preset.test_agent()

    if not args.generate_gif:
        returns = test_independent(env, agent, args.frames, side=args.agent)
        agent_names = ["first_0", "second_0", "third_0", "fourth_0"]
        with open("./outfiles/" + args.checkpoint + "_" + args.agent + ".txt",'w') as out:
            out.write(f"Environment: {args.env}\n")
            out.write(f"Checkpoint Name: {args.checkpoint}\n")
            out.write(f"Agent: {args.agent}\n")
            out.write(f"Returns: {str(returns)}\n")
            out.write(f"Average return: {str(np.mean(returns))}\n")
            out.write(f"Evaluation frames: {args.frames}\n")

        #print(returns_agent1(returns))
        # preset = independent({agent:base_builder for agent in env.agents}).env(env).hyperparameters(replay_buffer_size=350000,replay_start_size=100)
        # agent = preset.agent()
    else:
        name = f"{args.env}_{args.agent}"
        folder = f"frames/{name}/"
        os.makedirs(folder,exist_ok=True)
        os.makedirs("gifs",exist_ok=True)
        generate_episode_gifs(env, agent, args.frames, folder, side=args.agent)

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
