import sys

filename = sys.argv[1]

reward_found = False
policy_reward_max_found = False
episodes_found = False
iterations_found = False
steps_found = False

with open(filename,'r') as f:
    for line in reversed(list(f)):
        if "episode_reward_mean" in line:
            mean_reward = float(line.split()[1])
            reward_found = True
        if "training_iteration" in line:
            total_iterations = int(line.split()[1])
            iterations_found = True
        if "episodes_total" in line:
            total_episodes = int(line.split()[1])
            episodes_found = True
        if "timesteps_total" in line:
            total_steps = int(line.split()[1])
            steps_found = True
        if "policy_0" in line and policy_reward_mean:
            policy_reward_max = float(line.split()[1])
            policy_reward_max_found = True
        if "policy_reward_mean" in line:
            policy_reward_mean = True
            continue
        if reward_found and episodes_found and iterations_found:
            break
        policy_reward_mean = False

print("Average Total Reward: {} after {} episodes, {} iterations, and {} million steps.".format(mean_reward, total_episodes, total_iterations, round(total_steps/1e6,1)))
if policy_reward_max_found:
    print("Policy Reward Max: {}".format(policy_reward_max))
