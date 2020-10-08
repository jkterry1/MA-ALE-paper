
import sys
from train_atari import AtariModel, get_env, make_env_creator
import os

from supersuit import clip_reward_v0, sticky_actions_v0, resize_v0
from supersuit import frame_skip_v0, frame_stack_v1, agent_indicator_v0
import numpy as np

if __name__ == "__main__":

    assert len(sys.argv) == 2, "Input the environment name"
    env_name = sys.argv[1].lower()

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

    results_path = os.path.join(train_path,"random_reward.txt")

    max_num_steps = 500000

    game_env = get_env(env_name)

    def env_creator(args):
        env = game_env.env(obs_type='grayscale_image')
        #env = clip_reward_v0(env, lower_bound=-1, upper_bound=1)
        env = sticky_actions_v0(env, repeat_action_probability=0.25)
        env = resize_v0(env, 84, 84)
        #env = color_reduction_v0(env, mode='full')
        env = frame_skip_v0(env, 4)
        env = frame_stack_v1(env, 4)
        env = agent_indicator_v0(env, type_only=False)
        return env
    env = (env_creator(0))
    total_rewards = dict(zip(env.agents, [[] for _ in range(env.num_agents)]))
    num_steps = 0
    while num_steps < max_num_steps:
        observation = env.reset()
        prev_actions = env.rewards
        prev_rewards = env.rewards
        rewards = dict(zip(env.agents, [[0] for _ in range(env.num_agents)]))
        done = False
        iteration = 0
        policy_agent = 'first_0'
        while not done and num_steps < max_num_steps:
            for _ in env.agents:
                #print(observation.shape)
                #imsave("./"+str(iteration)+".png",observation[:,:,0])
                #env.render()
                observation = env.observe(env.agent_selection)
                action = env.action_spaces[policy_agent].sample() #same action space for all agents
                # action, _, _ = RLAgent.get_policy("policy_0").compute_single_action(observation, prev_action=prev_actions[env.agent_selection], prev_reward=prev_rewards[env.agent_selection])

                #print('Agent: {}, action: {}'.format(env.agent_selection,action))
                prev_actions[env.agent_selection] = action
                env.step(action, observe=False)
                #print('reward: {}, done: {}'.format(env.rewards, env.dones))
            prev_rewards = env.rewards
            for agent in env.agents:
                rewards[agent].append(prev_rewards[agent])
            done = any(env.dones.values())
            iteration += 1
            num_steps += 1
            if iteration > 10000:
                break
        for agent in env.agents:
            total_rewards[agent].append(np.sum(rewards[agent]))
        for agent in env.agents:
            print("Agent: {}, Reward: {}".format(agent, np.mean(rewards[agent])))
        print('Total reward: {}'.format(total_rewards))

    out_stat_fname = results_path
    mean_rew = np.mean(total_rewards[policy_agent])
    open(out_stat_fname,'w').write(str(mean_rew))
