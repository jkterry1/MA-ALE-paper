import os
from concurrent.futures import ThreadPoolExecutor
import subprocess
import multiprocessing as mp



all_envs = next(os.walk("/home/ben/ray_results_atari_baselines"))[1]

#done_envs = ['double_dunk', 'wizard_of_wor']
done_envs = []

all_run_args = []

for env in all_envs:
    if env in done_envs:
        continue
    run_args = (f"python play_atari_video.py {env} 2000")
    all_run_args.append(run_args.split())

executor = ThreadPoolExecutor(max_workers=int(3))                                                             
executor.map(subprocess.run,all_run_args)

