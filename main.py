import numpy as np
import random
import json

import utils
from consts import *
from env import GameEnv
from players import Player, UserPlayer, RLPlayer


def run_game(num_decks=1, num_players=3, mode="lord", players=[UserPlayer()], moveset=MOVESET_1, num_episodes=1, seed=None):
    if seed is not None:
        random.seed(seed)

    # Initialize the players and environment
    num_players, players = utils.adjust_player_count(num_decks, num_players, mode, players)
    env = GameEnv(num_decks, num_players, mode, moveset)
    
    for player in players:
        player.num_players = num_players
        player.deck_moves = env.deck_moves

    for episode in range(num_episodes):
        # Reset all player and environment variables
        for player in players:
            player.reset()
        env.reset(players)
        
        # Randomize the order that players play in each round
        random.shuffle(players)
        curr_player = 0
        
        # Deal the non-landlord cards by splitting the deck evenly
        landlord_cards, player_hands = utils.deal_regular_cards(num_players, num_decks, mode)
        for p in range(num_players):
            players[p].hand = player_hands[p]
            
        transitions = []
            
        # Players take turns to claim the landlord cards, starting from player 0
        if mode == "lord":
            for idx, player in enumerate(players):
                state = env.get_state(players, idx)
                pattern, leading_rank, choice, remainder = player.select_action(state, landlord_cards)
                action, new_state, reward, done = env.step(players, idx, pattern, leading_rank, choice, remainder, landlord_cards)
                
                # Record state transitions
                transitions.append((state, action, new_state, reward, done))
                
                # Stop when the landlord has been claimed/assigned
                if env.landlord_idx is not None:
                    curr_player = env.landlord_idx
                    break
        
        # Set the first player to be free to move
        players[curr_player].free = True
        
        utils.print_game(pattern, leading_rank, choice, new_state, players, curr_player, start=True, verbose=True)
        
        # Keep playing until the game is guaranteed to end
        done = False
        while not done:
            state = env.get_state(players, curr_player)
            pattern, leading_rank, choice, remainder = players[curr_player].select_action(state)
            action, new_state, reward, done = env.step(players, curr_player, pattern, leading_rank, choice, remainder, None)

            # Record state transitions
            transitions.append((state, action, new_state, reward, done))
            
            utils.print_game(pattern, leading_rank, choice, new_state, players, curr_player, start=False, verbose=True)
            
            if done:
                utils.announce_winner(mode, curr_player, players[curr_player].landlord)

            # Continue to the next player
            curr_player = (curr_player + 1) % num_players
        
        # TODO: Finalize rewards in the transition list after the game ends
        # TODO: Update RL agents using the game history (if training)

run_game()