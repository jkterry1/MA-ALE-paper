import os

all_envs = []

for f in os.listdir('./'):
    if f.endswith('.csv'):
        all_envs.append(f.split('.')[0])

print(all_envs)

for env in all_envs:
    os.system(f"python plot.py {env}")


