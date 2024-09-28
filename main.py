import numpy as np
import random
import json

import utils
from consts import *
from env import GameEnv
from players import DefaultPlayer, UserPlayer, RLPlayer


def run_game(num_decks=2, num_players=3, mode="lord", players=[], moveset=MOVESET_1, num_episodes=1, seed=None):
    if seed is not None:
        random.seed(seed)

    # Initialize the players and environment
    num_players, players = utils.adjust_player_count()
    env = GameEnv(num_decks, num_players, mode, moveset)
    
    for player in players:
        player.num_players = num_players
        player.deck_moves = env.deck_moves

    for episode in range(num_episodes):
        # TODO: Reset all players
        for player in players:
            player.reset()
        
        # Randomize the order that players play in each round
        random.shuffle(players)
        curr_player = 0
        
        # Deal the non-landlord cards by splitting the deck evenly
        landlord_cards, player_hands = utils.deal_regular_cards(num_players, num_decks, mode)
        for p in range(num_players):
            players[p].hand = player_hands[p]
        
        # Get the initial state right after reset
        state = env.reset(players)
            
        # Players take turns to claim the landlord cards, starting from player 0
        if mode == "lord":
            for idx, player in enumerate(players):
                pattern, choice, leading_rank, remainder = player.select_action(state)
                new_state, reward, done = env.step(players, curr_player, pattern, choice, leading_rank, remainder, landlord_cards)
                
                # TODO: Record state transitions here too
                
                # Stop when the landlord has been claimed/assigned
                if env.landlord_idx is not None:
                    curr_player = env.landlord_idx
                    break
        
        # Set the first player to be free to move
        players[curr_player].free = True
        
        # Keep playing until the game is guaranteed to end
        done = False
        while not done:
            # TODO: Move get_state into env; call it before select_action
            state = new_state
            
            pattern, choice, leading_rank, remainder = players[curr_player].select_action(state)
            new_state, reward, done = env.step(players, curr_player, pattern, choice, leading_rank, remainder)

            # TODO: Replay buffer and record transition stuff (if training)
            # transitions.append((state['agent_1'], action, reward, new_state['agent_1'], done))

            # Continue to the next player
            curr_player = (curr_player + 1) % num_players

        utils.announce_winner(mode, curr_player, player.curr_player.landlord)
        
        # TODO: Finalize rewards in the transition list after the game ends
        # TODO: Update RL agents using the game history (if training)
