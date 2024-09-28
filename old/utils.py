import os
import json
import random
import numpy as np
import itertools

from consts import *
    
# Print the current state of the game
def print_game(valid_move, pattern, prev_choice, leading_rank, all_remaining, skip_count, next_player, players, start=False, verbose=False):
    if valid_move:
        # Print the initial hands at the start of the game
        if start:
            if verbose:
                for idx, player in enumerate(players):
                    print(f"Player {idx}: {player.hand} {sum(player.hand)} {type(player).__name__} ({'Landlord' if player.landlord else 'Peasant'})")
            else:
                for idx, player in enumerate(players):
                    print(f"Player {idx}: {sum(player.hand)} remaining ({'Landlord' if player.landlord else 'Peasant'})")
                
        # Print the most recent move and new hands
        else:
            if verbose:
                print(f"Choice: [{freq_array_to_card_str(prev_choice)}], pattern: {pattern}, rank: {leading_rank}, card: {CARDS[leading_rank]}\n")
                for idx, player in enumerate(players):
                    print(f"Player {idx}: {player.hand} {sum(player.hand)} ({'Landlord' if player.landlord else 'Peasant'})")
            else:
                print(f"Choice: [{freq_array_to_card_str(prev_choice)}], pattern: {pattern}\n")
                for idx, player in enumerate(players):
                    print(f"Player {idx}: {sum(player.hand)} remaining ({'Landlord' if player.landlord else 'Peasant'})")
                    
        print(f"All remaining: {freq_array_to_card_str(all_remaining)}")
        print()
    else:
        print(f"Skip. Skip count: {skip_count}\n")
    
    # Announce the new current player
    print(f"Next player: {next_player}\n")


def finalize_rewards(mode, num_players, episode_transitions, winner, landlord):
    """
    Update the rewards for all players after the game ends, directly modifying episode_transitions
    """
    # In individual mode, one winner takes all
    if mode == "indv":
        for p in range(num_players):
            state, action, reward, next_state, done = episode_transitions[-p-1]
            curr_player = state["self"]["id"]
            if curr_player == winner:
                updated_reward = REWARDS["win"] * (num_players - 1)
            else:
                updated_reward = REWARDS["loss"]
            
            # Update the transition in place with the final reward
            episode_transitions[-p-1] = (state, action, updated_reward, next_state, done)
    
    # In landlord mode, peasants win if any peasant wins
    else:
        peasants = [p for p in range(num_players) if p != landlord]
        if winner == landlord:
            winners = [landlord]
            win_reward = REWARDS["win"] * (num_players - 1)
            loss_reward = REWARDS["loss"]
        else:
            winners = peasants
            win_reward = REWARDS["win"]
            loss_reward = REWARDS["loss"] * (num_players - 1)
            
        for p in range(num_players):
            state, action, reward, next_state, done = episode_transitions[-p-1]
            curr_player = state["self"]["id"]
            
            if curr_player in winners:
                updated_reward = win_reward
            else:
                updated_reward = loss_reward
            
            # Update the transition in place with the final reward
            episode_transitions[-p-1] = (state, action, updated_reward, next_state, done)
            
