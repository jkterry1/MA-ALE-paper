import sys

filename = sys.argv[1]

reward_found = False
episodes_found = False
iterations_found = False

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
        if reward_found and episodes_found and iterations_found:
            break

print("Average Total Reward: {} after {} episodes and {} iterations.".format(mean_reward, total_episodes, total_iterations))
