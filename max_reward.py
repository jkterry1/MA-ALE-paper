
import os
import numpy as np
import pandas as pd

import sys

env_name = sys.argv[1]
algorithm = sys.argv[2].upper()

#algorithm = algorithm.lower()

data_path = "../ray_results_atari/"+env_name+"/"+algorithm

df = pd.read_csv(os.path.join(data_path,'progress.csv'))
df = df[['training_iteration',"episode_reward_min","episode_reward_max","episode_reward_mean", "episodes_total"]]

iter_range = list(range(100,500000,100))
df2 = df[df['training_iteration'].isin(iter_range)]
iter_min = df2.loc[df2['episode_reward_min'].idxmax(), ['training_iteration', "episode_reward_min", "episodes_total"]]
iter_max = df2.loc[df2['episode_reward_max'].idxmax(), ['training_iteration', "episode_reward_max", "episodes_total"]]
iter_mean = df2.loc[df2['episode_reward_mean'].idxmax(), ['training_iteration', "episode_reward_mean", "episodes_total"]]

rew_min  = df['episode_reward_min'].max()
rew_max  = df['episode_reward_max'].max()
rew_mean = df['episode_reward_mean'].max()
epi_min  = df.loc[df['episode_reward_min'].idxmax(), ['episodes_total','training_iteration']]
epi_max  = df.loc[df['episode_reward_max'].idxmax(), ['episodes_total','training_iteration']]
epi_mean = df.loc[df['episode_reward_mean'].idxmax(), ['episodes_total','training_iteration']]

print("\n")
print("Min of ",rew_min, " at ", int(epi_min[0]), " episodes (",int(epi_min[1]),' iterations)')
print("Mean of ",rew_mean, " at ", int(epi_mean[0]), " episodes (",int(epi_mean[1]),' iterations)')
print("Max of ",rew_max, " at ", int(epi_max[0]), " episodes (",int(epi_max[1]),' iterations)')
print("\n")
print("Max Possible Min Reward of {} at {} episodes ({} iterations)".format(iter_min[1],int(iter_min[2]),int(iter_min[0])))
print("Max Possible Mean Reward of {} at {} episodes ({} iterations)".format(iter_mean[1],int(iter_mean[2]),int(iter_mean[0])))
print("Max Possible Max Reward of {} at {} episodes ({} iterations)\n\n".format(iter_max[1],int(iter_max[2]),int(iter_max[0])))
