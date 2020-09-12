
import os
import numpy as np
import pandas as pd

import sys

env_name = sys.argv[1]
algorithm = sys.argv[2].upper()

#algorithm = algorithm.lower()

data_path = "../ray_results_base/"+env_name+"/"+algorithm

df = pd.read_csv(os.path.join(data_path,'progress.csv'))
df = df[['training_iteration', "episode_reward_mean", "episodes_total"]]

iter_range = list(range(10,500000,10))
df2 = df[df['training_iteration'].isin(iter_range)]
iter_max = df2.loc[df2['episode_reward_mean'].idxmax(), ['training_iteration', "episode_reward_mean", "episodes_total"]]


rew_max = df['episode_reward_mean'].max()
epi_max = df.loc[df['episode_reward_mean'].idxmax(), ['episodes_total','training_iteration']]

print("Max of ",rew_max, " at ", int(epi_max[0]), " episodes (",int(epi_max[1]),' iterations)')

print("Max Possible Reward of {} at {} episodes ({} iterations)\n\n".format(iter_max[1],int(iter_max[2]),int(iter_max[0])))
