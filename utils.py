import os
import json
import random
import numpy as np
import itertools
from consts import *

def smallest_valid_choice(hand, pattern, prev_choice=None, leading_rank=-1):
    """
    Choose the smallest playable hand given a pattern
    
    params:
        hand: Numpy frequency array like [1 1 1 3 0 1 1 2 0 2 1 0 1 0 0]
        pattern: String name of the pattern like "3x1+2"
        
    returns:
        valid_move: True or False for whether the hand contains the given pattern
        choice: The smallest set of cards that satisfy the pattern
    """
    valid_move = False
    choice = [0] * 15
    
    match pattern:
        # Lead is the lowest rank
        case "1x5":
            singles_count = 0
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 1:
                    singles_count += 1
                else:
                    singles_count = 0
                    
                if singles_count == 5:
                    valid_move = True
                    for i in range(5):
                        choice[rank-i] = 1
                    leading_rank = rank - 4
                    break
        
        # Lead is the lowest rank
        case "2x3":
            doubles_count = 0
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 2:
                    doubles_count += 1
                else:
                    doubles_count = 0
                    
                if doubles_count == 3:
                    valid_move = True
                    for i in range(3):
                        choice[rank-i] = 2
                    leading_rank = rank - 2
                    break
                
        case "3x1+1":
            triple_rank = -1
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 3:
                    triple_rank = rank
                    break
            if triple_rank >= 0:
                for rank, freq in enumerate(hand):
                    if freq >= 1 and rank != triple_rank:
                        valid_move = True
                        choice[triple_rank] = 3
                        choice[rank] = 1
                        leading_rank = triple_rank
                        break
        
        case "3x1+2":
            triple_rank = -1
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 3:
                    triple_rank = rank
                    break
            if triple_rank >= 0:
                for rank, freq in enumerate(hand):
                    if freq >= 2 and rank != triple_rank:
                        valid_move = True
                        choice[triple_rank] = 3
                        choice[rank] = 2
                        leading_rank = triple_rank
                        break
        
        case "1":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 1:
                    valid_move = True
                    choice[rank] = 1
                    leading_rank = rank
                    break
        
        case "2":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 2:
                    valid_move = True
                    choice[rank] = 2
                    leading_rank = rank
                    break
        
        case "3":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 3:
                    valid_move = True
                    choice[rank] = 3
                    leading_rank = rank
                    break
                
        case "4":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 4:
                    valid_move = True
                    choice[rank] = 4
                    leading_rank = rank
                    break
        
        case "5":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 5:
                    valid_move = True
                    choice[rank] = 5
                    leading_rank = rank
                    break
        
        case "6":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 6:
                    valid_move = True
                    choice[rank] = 6
                    leading_rank = rank
                    break
        
        case "7":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 7:
                    valid_move = True
                    choice[rank] = 7
                    leading_rank = rank
                    break
                
        case "8":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 8:
                    valid_move = True
                    choice[rank] = 8
                    leading_rank = rank
                    break
                
        # Small bomb, already handled later with card bomb
        case "4.5":
            pass
        
        # Big bomb, must skip turn
        case "8.5":
            pass
            
        case _:
            raise NotImplementedError("Pattern not available")
        
    # If can't play any moves, check for card bombs
    if not valid_move:
        # If the previous pattern was not a bomb, any bomb can be played
        if pattern not in ["4", "5", "6", "7", "8", "4.5", "8.5"]:
            for rank, freq in enumerate(hand):
                if freq >= 4:
                    valid_move = True
                    choice[rank] = freq     # Bomb with all cards of that rank
                    leading_rank = rank
                    pattern = str(freq)     # ex. "5"
                    break
        
        # Else only bombs with bigger number or more cards can be played
        else:
            for rank, freq in enumerate(hand):
                if (freq == float(pattern) and rank > leading_rank) or freq > float(pattern):
                    valid_move = True
                    choice[rank] = freq
                    leading_rank = rank
                    pattern = str(freq)
                    break
                
    # If there are no card bombs, check for Joker bombs
    if not valid_move:
        # Small Joker bomb can only override regular cards and 4-bombs
        if pattern not in ["5", "6", "7", "8", "4.5"]:
            if hand[13] == 1 and hand[14] == 1:
                valid_move = True
                choice[13] = 1
                choice[14] = 1
                leading_rank = 13
                pattern = "4.5"
                
        # Big Joker bomb can override all cards and bombs
        else:
            if hand[13] == 2 and hand[14] == 2:
                valid_move = True
                choice[13] = 2
                choice[14] = 2
                leading_rank = 13
                pattern = "8.5"
    
    return valid_move, pattern, np.array(choice), leading_rank


# Convert frequency array to card string
def freq_array_to_card_str(hand):
    card_str = ""
    for rank, freq in enumerate(hand):
        card_str += CARDS[rank] * freq + " "
    return card_str
    
    
def read_user_cards(user_cards, pattern, leading_rank, hand):
    """
    Convert card string to frequency array.
    
    params:
        user_cards: String of cards like "334455"
        pattern: String name of the pattern like "3x1+2"
        leading_rank: The previous leading rank
    """
    if not user_cards:  # User chose to skip the turn
        return False, pattern, None, -1, True
    
    else:   # Did not skip turn
        # Note: If free to move, the leading rank is already reset to -1 in main.py
        choice = [0] * 15
        # Check if the user played a bomb then update pattern and choice
        # If length = 1, 2, or 3, it is a regular play and the pattern can only be set by the player if the player was free to move
        # If length >= 4, it is a bomb that can overwrite the pattern
        if user_cards[0] == user_cards[-1] and len(user_cards) >= 4:
            new_pattern = len(user_cards)
            if pattern not in BOMB_SET or new_pattern > float(pattern):
                pattern = str(new_pattern)
                leading_rank = -1
            choice[RANKS[user_cards[0]]] = len(user_cards)
        elif user_cards == "SL":
            if pattern not in BOMB_SET or 4.5 > float(pattern):
                pattern = "4.5"
                leading_rank = -1
            choice[13] = 1
            choice[14] = 1
        elif user_cards == "SSLL":
            if pattern not in BOMB_SET or 8.5 > float(pattern):
                pattern = "8.5"
                leading_rank = -1
            choice[13] = 2
            choice[14] = 2
        # If no bomb, just count cards assuming the previous pattern is unchanged
        else:
            for c in user_cards:
                choice[RANKS[c]] += 1
                
        return True, pattern, choice, RANKS[user_cards[0]], is_user_choice_valid(pattern, choice, user_cards, leading_rank, hand)
    
    
def is_user_choice_valid(pattern, choice, user_cards, leading_rank, hand):
    """
    Check if the user input card string matches the pattern
    Currently covers all patterns in the simple moveset
    Assume leading_rank is set to -1 if a new pattern is chosen (ex. free, bomb)
    leading_rank only matters if the previous pattern continues
    
    params:
        pattern: String name of the pattern like "3x1+2"
        choice: Numpy frequency array like [1 1 1 3 0 1 1 2 0 2 1 0 1 0 0]
        leading_rank: The previous leading rank
    """
    ranks = [RANKS[c] for c in user_cards]
    
    # New leading rank must be higher than the previous leading rank
    if RANKS[user_cards[0]] <= leading_rank:
        return False
    
    # All selected cards must be in the player's hand
    new_hand = hand - choice
    if min(new_hand) < 0:
        return False
    
    match pattern:
        case "1x5":
            if sum(choice) != 5 or set(choice) != {0, 1}:   # 5 singular cards
                return False
            if max(ranks) - min(ranks) != 4:                # Consecutive ranks
                return False
            
        case "2x3":
            if sum(choice) != 6 or set(choice) != {0, 2}:   # 3 pairs of cards
                return False
            if max(ranks) - min(ranks) != 2:                # Consecutive ranks
                return False
            
        case "3x1+1":
            if sum(choice) != 4 or set(choice) != {0, 1, 3}:    # 1 triple and 1 singular card
                return False
            
        case "3x1+2":
            if sum(choice) != 5 or set(choice) != {0, 2, 3}:    # 1 triple and 1 pair of cards
                return False
            
        case "1":
            if sum(choice) != 1 or set(choice) != {0, 1}:       # 1 singular card
                return False
            
        case "2":
            if sum(choice) != 2 or set(choice) != {0, 2}:       # 1 pair of cards
                return False
            
        case "3":
            if sum(choice) != 3 or set(choice) != {0, 3}:       # 1 triple
                return False
            
        case "4":
            if sum(choice) != 4 or set(choice) != {0, 4}:       # 1 4-card bomb
                return False
            
        case "5":
            if sum(choice) != 5 or set(choice) != {0, 5}:       # 1 5-card bomb
                return False
            
        case "6":
            if sum(choice) != 6 or set(choice) != {0, 6}:       # 1 6-card bomb
                return False
            
        case "7":
            if sum(choice) != 7 or set(choice) != {0, 7}:       # 1 7-card bomb
                return False
            
        case "8":
            if sum(choice) != 8 or set(choice) != {0, 8}:       # 1 8-card bomb
                return False
            
        # Note: Joker bombs were hardcoded in read_user_cards(), no need to check here
        case "4.5":
            pass
        
        case "8.5":
            pass
            
        case _:
            raise NotImplementedError("Pattern not available")
        
    return True


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
            
# Generate all card combinations for all patterns in the moveset
# Read from json if available, else generate and save to json
def get_all_possible_moves(filename="data/all_possible_moves.json", overwrite=False):
    if os.path.exists(filename) and not overwrite:
        with open(filename, "r") as f:
            moves = json.load(f)
        return moves
    
    # Structure: (pattern, leading rank, card frequency array)
    moves = []
    ranks = list(range(15))
    
    # 1xn: n consecutive singles (5 --> 15)
    for n in range(5, 16):
        for i in range(16-n):
            moves.append((f"1x{n}", i, [0]*i + [1]*n + [0]*(15-i-n)))
            
    # 2xn: n consecutive pairs (3 --> 15)
    for n in range(3, 16):
        for i in range(16-n):
            moves.append((f"2x{n}", i, [0]*i + [2]*n + [0]*(15-i-n)))
    
    # 3xn+1: n consecutive triples + n singles (1 --> 2)
    for n in range(1, 3):
        for i in range(14-n):
            i_move = [0]*i + [3]*n + [0]*(15-i-n)

            # Find all combinations of n random singles
            for singles in itertools.combinations_with_replacement(ranks, n):
                move_copy = i_move[:]
                for single_rank in singles:
                    move_copy[single_rank] += 1
                moves.append((f"3x{n}+1", i, move_copy))
                
    # 3xn+2: n consecutive triples + n pairs (1 --> 2)
    for n in range(1, 3):
        for i in range(14-n):
            i_move = [0]*i + [3]*n + [0]*(15-i-n)

            # Find all combinations of n random pairs
            for pairs in itertools.combinations_with_replacement(ranks, n):
                move_copy = i_move[:]
                for pair_rank in pairs:
                    move_copy[pair_rank] += 2
                    
                if move_copy[13] <= 2 and move_copy[14] <= 2:
                    moves.append((f"3x{n}+2", i, move_copy))
        
    # n: n of a kind (1 --> 8)
    # Note: Jokers can only have 1 or 2 of a kind
    for n in range(1, 9):
        if n <= 2:
            for i in range(15):
                moves.append((str(n), i, [0]*i + [n] + [0]*(14-i)))
        else:
            for i in range(13):
                moves.append((str(n), i, [0]*i + [n] + [0]*(14-i)))
                
    # Small and large Joker bombs
    moves.append(("4.5", 13, [0]*13 + [1, 1]))
    moves.append(("8.5", 13, [0]*13 + [2, 2]))
            
    with open(filename, "w") as f:
        json.dump(moves, f)
    return moves
            
# Filter all possible moves to those possible with the given deck
# For example, in games with 1 deck, 8-bombs are impossible
def deck_possible_moves(moves, cards_remaining, moveset):
    valid_moves = []
    for move in moves:
        pattern, _, move_freq = move
        if pattern in moveset and min(cards_remaining - move_freq) >= 0:
            valid_moves.append(move)
    return valid_moves
            

# Return playable moves, if any
# If yes, return True and a boolean mask of all possible moves
# Else, return False and None
def hand_possible_moves(pattern, prev_choice, leading_rank, hand, prev_hand_valid, moveset):
    # TODO: Store the list of moves previously found to be valid, based only on the hand
    # Each time, only incrementally check this list
    # Then return the hand-valid + play-valid moves
    # At the start, check all moves (prev_hand_valid = all_possible_moves)
    pass
