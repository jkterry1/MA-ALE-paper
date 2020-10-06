import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from train_atari import AtariModel, get_env, make_env_creator
from pettingzooenv import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env, register_trainable
from ray import tune
import gym
import random
import numpy as np
import ray
from pettingzoo.utils.to_parallel import from_parallel
import os
import sys
import pickle

from ray.rllib.rollout import rollout, keep_going, DefaultMapping
from ray.rllib.agents.dqn import ApexTrainer

if __name__ == "__main__":
    methods = ["ADQN", "PPO", "RDQN"]

    assert len(sys.argv) == 4, "Input the environment name, data_path, checkpoint_num"
    env_name = sys.argv[1].lower()
    data_path = sys.argv[2]
    checkpoint_num = sys.argv[3]
    method = "ADQN"
    assert method in methods, "Method should be one of {}".format(methods)

    checkpoint_path = f"{data_path}/checkpoint_{checkpoint_num}/checkpoint-{checkpoint_num}"

    Trainer = ApexTrainer

    game_env = get_env(env_name)
    env_creator = make_env_creator(game_env, clip_rewards=False)

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    test_env = ParallelPettingZooEnv(env_creator({}))
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    ModelCatalog.register_custom_model("AtariModel", AtariModel)

    def gen_policy(i):
        config = {
            "model": {
                "custom_model": "AtariModel",
            },
            "gamma": 0.99,
        }
        return (None, obs_space, act_space, config)
    policies = {"policy_0": gen_policy(0)}

    config_path = os.path.join(data_path, "params.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    config['num_gpus']=0
    config['num_workers']=0
    # ray.init()

    results_path = os.path.join(data_path,"checkpoint_values")
    os.makedirs(results_path,exist_ok=True)
    result_path = os.path.join(results_path,f"checkpoint{checkpoint_num}.txt")

    ray.init(num_cpus=0,num_gpus=0)

    RLAgent = Trainer(env=env_name, config=config)
    RLAgent.restore(checkpoint_path)


    env = from_parallel(env_creator(0))
    observation = env.reset()
    actions = env.rewards
    rewardss = env.rewards
    rewards = [0]
    rewards2 = [0]
    total_reward = 0
    total_reward2 = 0
    done = False
    iteration = 0
    policy_agent = 'first_0'
    policy =  RLAgent.get_policy("policy_0")

    max_num_steps = 50000
    num_steps = 0
    num_episodes = 0
    while num_steps < max_num_steps:
        while not done and num_steps < max_num_steps:
            for _ in env.agents:
                #print(observation.shape)
                #imsave("./"+str(iteration)+".png",observation[:,:,0])
                # env.render()
                observation = env.observe(env.agent_selection)
                ####
                if env.agent_selection == policy_agent:
                    action, _, _ = policy.compute_single_action(observation, prev_action=actions[env.agent_selection], prev_reward=rewardss[env.agent_selection])
                else:
                    action = env.action_spaces[policy_agent].sample() #same action space for all agents
                ####
                #action, _, _ = RLAgent.get_policy("policy_0").compute_single_action(observation, prev_action=actions[env.agent_selection] , prev_reward=rewardss[env.agent_selection])
                ####
                # print('Agent: {}, action: {}'.format(env.agent_selection,action))
                actions[env.agent_selection] = action
                env.step(action, observe=False)
                # print('reward: {}, done: {}'.format(env.rewards, env.dones))
            rewardss = env.rewards
            reward = env.rewards['first_0']
            reward2 = env.rewards['second_0']
            rewards.append(reward)
            rewards2.append(env.rewards['second_0'])
            done = any(env.dones.values())
            total_reward += reward
            total_reward2 += reward2
            iteration += 1
            num_steps += 1
            if num_steps % 1000 == 0:
                print(f"{num_steps/max_num_steps}% done, average reward is: {sum(rewards)/(num_episodes+1)}")

        done = False
        observation = env.reset()
        num_episodes += 1

    out_stat_fname = result_path
    mean_rew = sum(rewards)/num_episodes
    open(out_stat_fname,'w').write(str(mean_rew))
