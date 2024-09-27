import numpy as np
import random
import json

import utils
from consts import *
from env import GameEnv
from players import DefaultPlayer, UserPlayer, RLPlayer

EPISODES = 1

num_decks = 2
num_players = 3
mode = "lord"
players = []
moveset = MOVESET_1
seed = None

# Initialize the players
num_players, players = utils.adjust_player_count()
for player in players:
    player.moveset = moveset
random.shuffle(players)

# Initialize the environment
env = GameEnv(num_decks, num_players, mode, players, moveset, seed)

for episode in range(EPISODES):
    # TODO: Reset all players
    for player in players:
        player.reset()
    
    # Deal the non-landlord cards by splitting the deck evenly
    landlord_cards, player_hands = utils.deal_regular_cards(num_players, num_decks, mode)
    for p in range(num_players):
        players[p].hand = player_hands[p]

    # TODO: Record landlord claims as a state
    if mode == "lord":
        # TODO: Deal the landlord cards, set landlord_idx, curr_player, player.landlord, and update landlord hand
        pass

    # Reset all environment variables
    # Note: Env receives players purely for recording purposes
    # Env doesn't affect player values in any way
    env.reset(players)

    # TODO: Set the first player to be free to move
    # players[curr_player].free = True
    
    # TODO: Make sure to get the initial state
    
    done = False
    while not done:
        # valid_move, pattern, prev_choice, leading_rank, remainder = players[curr_player].select_action(state=curr_state)
        # new_state, reward, done = env.step(players, curr_player, valid_move, pattern, choice, leading_rank, remainder)

        # TODO: Replay buffer and record transition stuff

        # # Continue to the next player
        # curr_player = (curr_player + 1) % self.num_players
        pass


    # utils.announce_winner(mode, curr_player, landlord_idx)
    # TODO: Finalize rewards after the game ends
    # TODO: Update RL agents using the game history
