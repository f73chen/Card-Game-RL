import numpy as np
import random
import torch
import json

import utils
from consts import *
from game_env import GameEnv
from players import DefaultPlayer, UserPlayer, RLPlayer
# from train import RNN_DQN


# Play and store the game
# env = GameEnv(num_decks=2, num_players=3, mode="lord", players=[])
# env.reset()
# history = env.play_game(verbose=False)

# game_name = "data/user_game.json"
# json.dump(history, open(game_name, "w"))

utils.generate_all_possible_moves()

# Load and replay the game
# history = json.load(open("data/user_game.json", "r"))
# env.replay(history)

# # Load the model for inference
# rnn_dqn = RNN_DQN(state_size, action_size)
# rnn_dqn.load_state_dict(torch.load('rnn_dqn_model.pth'))
# rnn_dqn.eval()

# # Example usage
# env = GameEnv(num_decks=1, num_players=3, mode="lord", players=[])
# env.play_with_model(rnn_dqn, num_games=10)