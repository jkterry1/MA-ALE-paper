import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy
from scipy import signal
import sys

matplotlib.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.size": 6,
    "legend.fontsize": 5,
    "ytick.labelsize": 4,
    "text.usetex": True,
    "pgf.rcfonts": False
});

env_name = sys.argv[1].lower()

plt.figure(figsize=(2.65, 1.5))
data_path = "./"

df = pd.read_csv(os.path.join(data_path, env_name+'.csv'))
df = df[['timesteps_total', "reward"]]
data = df.to_numpy()
#filtered = scipy.signal.savgol_filter(data[:, 1], int(len(data[:, 1])/110)+2, 5)
filtered = data[:,1]
plt.plot(data[:, 0], filtered, label=env_name, linewidth=0.6, color='tab:blue', linestyle='-')

plt.xlabel('Episode', labelpad=1)
plt.ylabel('Average Total Reward', labelpad=1)
plt.title(env_name.capitalize())
#plt.xticks(ticks=[10000,20000,30000,40000,50000],labels=['10k','20k','30k','40k','50k'])
#plt.xlim(0, 60000)
#plt.yticks(ticks=[0,150,300,450,600],labels=['0','150','300','450','600'])
#plt.ylim(-150, 750)
plt.tight_layout()
#plt.legend(loc='lower center', ncol=2, labelspacing=.2, columnspacing=.25, borderpad=.25, bbox_to_anchor=(0.5, -0.6))
plt.margins(x=0)
#plt.savefig("qmix_pursuit.pgf", bbox_inches = 'tight',pad_inches = .025)
plt.savefig(env_name+".png", bbox_inches = 'tight',pad_inches = .025, dpi=600)


