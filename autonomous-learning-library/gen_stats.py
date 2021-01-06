import pandas
import numpy as np
data = pandas.read_csv("arg.csv")
#print(data)
games = set(data['game'])
for game in games:
    game_data = data[(data['game'] == game)]
    print(game, np.mean(game_data['reward']))
