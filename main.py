import numpy as np
import random
import pickle
import torch

import utils
from consts import *
from game_env import GameEnv
from players import DefaultPlayer, UserPlayer, RLPlayer
# from train import RNN_DQN


env = GameEnv(num_decks=1, num_players=3, mode="lord", players=[UserPlayer()])
env.reset()
history = env.play_game(verbose=False)

# game_name = "user_game_0.pkl"
# pickle.dump(history, open(game_name, "wb"))
# history = pickle.load(open(game_name, "rb"))
# env.replay(history)

# # Load the model for inference
# rnn_dqn = RNN_DQN(state_size, action_size)
# rnn_dqn.load_state_dict(torch.load('rnn_dqn_model.pth'))
# rnn_dqn.eval()

# # Example usage
# env = GameEnv(num_decks=1, num_players=3, mode="lord", players=[])
# env.play_with_model(rnn_dqn, num_games=10)