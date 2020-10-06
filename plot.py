
import os
import numpy as np
import pandas as pd

import sys

env_name = sys.argv[1]
folder_name = sys.argv[2]
algorithm = "ADQN"

#algorithm = algorithm.lower()

data_path = f"../ray_results_atari_baselines/{env_name}/{algorithm}/{folder_name}"

df = pd.read_csv(os.path.join(data_path,'progress.csv'))
df = df[['training_iteration',"timesteps_total"]]

#iter_range = list(range(20,500000,20))
#df2 = df[df['training_iteration'].isin(iter_range)]

rewards = [0,0]

data_path = f"../ray_results_atari_baselines/{env_name}/{algorithm}/{folder_name}/checkpoint_values/"
for file_name in os.listdir(data_path):
    if file_name.endswith(".txt"):
        iteration = int(file_name.split('checkpoint')[1].split('.txt')[0])
        with open(data_path+file_name,'r') as fp:
            for line in fp:
                reward = float(line)
        df.loc[df['training_iteration'] == iteration, "reward"] = reward

df.dropna(subset=['reward'], inplace=True)


df.to_csv(env_name+'.csv', index=False)
