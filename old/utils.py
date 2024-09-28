import os
import json
import random
import numpy as np
import itertools

from consts import *

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
            
