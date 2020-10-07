import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'

from train_atari import AtariModel, get_env, make_env_creator
from pettingzooenv import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env, register_trainable
from ray import tune
import gym
import random
import numpy as np
import ray
import os
import sys
import multiprocessing as mp
import pickle
import subprocess

from ray.rllib.rollout import rollout, keep_going, DefaultMapping
from ray.rllib.agents.dqn import ApexTrainer
from concurrent.futures import ThreadPoolExecutor


if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    methods = ["ADQN", "PPO", "RDQN"]

    assert len(sys.argv) == 3, "Input the environment name, num parallel jobs, is self_play"
    env_name = sys.argv[1].lower()
    num_parallel_jobs = int(sys.argv[2])
    method = "ADQN"
    assert method in methods, "Method should be one of {}".format(methods)

    Trainer = ApexTrainer

    game_env = get_env(env_name)

    parent_save_path = os.path.expanduser("~/ray_results_atari_baselines/"+env_name+"/")
    print(list(os.listdir(parent_save_path)))
    data_paths = [dir for dir in os.listdir(parent_save_path) if os.path.isdir(os.path.join(parent_save_path,dir))]
    assert len(data_paths) > 0, f"there are no ray results for environment {env_name}"
    assert len(data_paths) < 2, f"there are too many ray results for environment {env_name}, results are ambigious. Please delete one of the results"
    algo_path = os.path.join(parent_save_path,data_paths[0])
    data_paths = [dir for dir in os.listdir(algo_path) if os.path.isdir(os.path.join(algo_path,dir))]
    assert len(data_paths) > 0, f"there are no ray results for environment {env_name}"
    assert len(data_paths) < 2, f"there are too many ray results for environment {env_name}, results are ambigious. Please delete one of the results"
    train_path = os.path.join(algo_path,data_paths[0])

    csv_results_path = os.path.join(train_path,"checkpoint_values.csv")
    #open(csv_results_path,'w').write("checkpoint,score")

    all_run_args = []
    for i in range(1,85+1):
        checkpoint_num = i*20*2
        checkpoint_path = f"{train_path}/checkpoint_{checkpoint_num}/checkpoint-{checkpoint_num}"

        run_args = (f"python collect_reward.py {env_name} {train_path} {checkpoint_num}")
        print(run_args)
        all_run_args.append(run_args.split())

        # RLAgent = Trainer(env=env_name, config=config)
        # RLAgent.restore(checkpoint_path)
        # num_steps = 200000
        #rollout(RLAgent, env_name, num_steps)

    executor = ThreadPoolExecutor(max_workers=num_parallel_jobs)
    executor.map(subprocess.run,all_run_args)
