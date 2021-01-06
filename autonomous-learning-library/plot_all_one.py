import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas
import os
import scipy
from scipy import signal
import sys

def get_env_name(env_name):
    env_name = env_name[:env_name.rfind("_")]
    if env_name=='boxing':
        game_env = 'Boxing'
    elif env_name=='combat_plane':
        game_env = 'Combat Plane'
    elif env_name=='combat_tank':
        game_env = 'Combat Tank'
    elif env_name=='double_dunk':
        game_env = 'Double Dunk'
    elif env_name=='entombed_competitive':
        game_env = 'Entombed Competitive'
    elif env_name=='entombed_cooperative':
        game_env = 'Entombed Cooperative'
    elif env_name=='flag_capture':
        game_env = 'Flag Capture'
    elif env_name=='ice_hockey':
        game_env = 'Ice Hockey'
    elif env_name=='joust':
        game_env = 'Joust'
    elif env_name=='mario_bros':
        game_env = 'Mario Bros'
    elif env_name=='maze_craze':
        game_env = 'Maze Craze'
    elif env_name=='othello':
        game_env = 'Othello'
    elif env_name=='basketball_pong':
        game_env = 'Basketball Pong'
    elif env_name=='pong':
        game_env = 'Pong'
    elif env_name=='foozpong':
        game_env = 'Foozpong'
    elif env_name=='quadrapong':
        game_env = 'Quadrapong'
    elif env_name=='volleyball_pong':
        game_env = 'Volleyball Pong'
    elif env_name=='space_invaders':
        game_env = 'Space Invaders'
    elif env_name=='space_war':
        game_env = 'Space War'
    elif env_name=='surround':
        game_env = 'Surround'
    elif env_name=='tennis':
        game_env = 'Tennis'
    elif env_name=='video_checkers':
        game_env = 'Video Checkers'
    elif env_name=='warlords':
        game_env = 'Warlords'
    elif env_name=='wizard_of_wor':
        game_env = 'Wizard of Wor'
    else:
        raise RuntimeError(f"{env_name} not found")
    return game_env

def main():

    assert len(sys.argv) == 2

    csv_name = sys.argv[1]

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

    plt.figure(figsize=(2.65*3*1.0, 1.5*7*1.0))

    csv_data = pandas.read_csv(csv_name)
    random_data = csv_data[csv_data['vs_random'] & csv_data['agent_random']]
    csv_data = csv_data[(csv_data['agent'] == "first_0") & ~csv_data['vs_random'] & ~csv_data['agent_random']]
    #print(data)
    all_envs = set(csv_data['game'])
    print(all_envs)

    all_env_names = {env: get_env_name(env) for env in all_envs}
    all_envs = sorted(all_envs, key=str.lower)

    plot_ind = 1
    for env in all_envs:
        print("plotted")
        df = csv_data[(csv_data['game'] == env)]
        plt.subplot(8,3,plot_ind)
        rand_reward = random_data[(random_data['game'] == env)].iloc[0]['agent1_rew']
        print(rand_reward)
        #df = pd.read_csv(os.path.join(data_path, env+'.csv'))
        df = df[['checkpoint', "agent1_rew"]]
        data = df.to_numpy()
        #filtered = scipy.signal.savgol_filter(data[:, 1], int(len(data[:, 1])/110)+2, 5)
        filtered = data[:,1]
        line, = plt.plot(data[:, 0], filtered, label=env, linewidth=0.6, color='#0530ad', linestyle='-')
        rand_line, = plt.plot(data[:, 0],rand_reward*np.ones_like(data[:, 0]), label=env, linewidth=0.6, color='#A0522D', linestyle='-')
        plt.xlabel('Steps', labelpad=1)
        plt.ylabel('Average Total Reward', labelpad=1)
        plt.title(all_env_names[env])
        #plt.xticks(ticks=[10000,20000,30000,40000,50000],labels=['10k','20k','30k','40k','50k'])
        #plt.xlim(0, 60000)
        #plt.yticks(ticks=[0,150,300,450,600],labels=['0','150','300','450','600'])
        #plt.ylim(-150, 750)
        plt.tight_layout()
        #plt.legend(loc='lower center', ncol=2, labelspacing=.2, columnspacing=.25, borderpad=.25, bbox_to_anchor=(0.5, -0.6))
        plt.margins(x=0)
        plot_ind += 1


    plt.figlegend([line,rand_line],['Trained Agent vs Random Agent', "Random Agent vs Random Agent"], fontsize='x-large', loc='lower center', ncol=1, labelspacing=.2, columnspacing=.25, borderpad=.25, bbox_to_anchor=(0.68,0.06))
    plt.savefig("atari_results.pgf", bbox_inches = 'tight',pad_inches = .025)
    plt.savefig("atari_results.png", bbox_inches = 'tight',pad_inches = .025, dpi=600)

if __name__ == "__main__":
    main()
