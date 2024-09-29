import os
import json
import random
import numpy as np
import itertools

from consts import *
from players import Player


def adjust_player_count(num_decks, num_players, mode, players):
    """
    Adjust the number of players and decks based on the win mode.
    """
    # Individual can have 3 or 4 players, landlord can only have 3 players
    if mode == "indv":
        if num_players < 3:
            num_players = 3
        elif num_players > 4:
            num_players = 4
    elif mode == "lord":
        num_players = 3
    
    # Number of decks can only be 1 or 2
    if num_decks < 1:
        num_decks = 1
    elif num_decks > 2:
        num_decks = 2
    
    # Remove extra players and add default players if necessary
    players = players[:num_players]
    for p in range(num_players - len(players)):
        players.append(Player())
    
    return num_players, players


def deal_regular_cards(num_players, num_decks, mode):
    card_freq = np.array(CARD_FREQ) * num_decks # Total number of cards in the deck(s)
    
    cards_per_player = CARDS_PER_PLAYER[mode][num_players][num_decks][0]
    hands = [np.array([0] * NUM_RANKS) for _ in range(num_players)] # Start with empty hands
    
    # Deal cards for each player
    for p in range(num_players):
        for card in range(cards_per_player[p]):
            dealt = False
            while not dealt:
                idx = random.randint(0, NUM_RANKS-1)
                if card_freq[idx] > 0:
                    card_freq[idx] -= 1
                    hands[p][idx] += 1
                    dealt = True
        
    return card_freq, hands

# Generate all card combinations for all patterns in the moveset
# Read from json if available, else generate and save to json
def get_all_moves(filename="data/all_moves.json", overwrite=False):
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
    
    moves.append(("claim_landlord", 0, [0]*15))
    moves.append(("refuse_landlord", 0, [0]*15))
    moves.append(("skip", 0, [0]*15))
            
    with open(filename, "w") as f:
        json.dump(moves, f)
    return moves
            
# Filter all possible moves to those possible with the given deck and moveset
# For example, in games with 1 deck, 8-bombs are impossible
def get_deck_moves(all_moves, cards_remaining, moveset):
    deck_moves = []
    for move in all_moves:
        pattern, _, move_freq = move
        if pattern in moveset and min(cards_remaining - move_freq) >= 0:
            deck_moves.append(move)
    return np.array(deck_moves, dtype=object)
            

def get_hand_moves(hand, free, prev_pattern, prev_leading_rank, hand_mask, deck_moves, choosing_landlord):
    """
    Return a mask of moves the player can play both for the current hand and the current situation
    
    params:
        hand: Numpy frequency array like [1 1 1 3 0 1 1 2 0 2 1 0 1 0 0]
        free: Whether the player is free to choose any move
        prev_pattern: The pattern of the previous move
        prev_leading_rank: The rank of the leading card in the previous move
        hand_mask: Mask of playable moves based on the entire hand
        deck_moves: List of all possible moves like (pattern, leading rank, card frequency array)
        
    returns:
        hand_mask: Updated mask of playable moves based on the current hand
        curr_mask: Mask of playable moves based on the current situation
    """
    if hand_mask is None:
        hand_mask = [True] * len(deck_moves)
    
    if choosing_landlord:
        curr_mask = [False] * len(deck_moves)
        curr_mask[-3] = True   # claim_landlord
        curr_mask[-2] = True   # refuse_landlord
        
    else:
        # In normal play, disable landlord claiming
        hand_mask[-3] = False
        hand_mask[-2] = False
    
        # Update the hand mask based on the new hand
        for i, mask in enumerate(hand_mask):
            if mask and min(hand - deck_moves[i][2]) < 0:
                hand_mask[i] = False

        # Next, check the specific situation based on free, pattern, and leading_rank
        # If player is free, any move from the hand is playable
        if free:
            curr_mask = hand_mask.copy()
        
        else:
            curr_mask = hand_mask.copy()
            for i, hand in enumerate(hand_mask):
                pattern, leading_rank, choice = deck_moves[i]
                
                # First, reject actions based on pattern and leading_rank
                if not hand or pattern != prev_pattern or leading_rank <= prev_leading_rank:
                    curr_mask[i] = False
                    
                # Then, make bombs available to play if the criteria is met
                if hand and pattern in BOMB_SET:
                    # Can always bomb if the previous move isn't a bomb
                    if prev_pattern not in BOMB_SET:
                        curr_mask[i] = True
                        
                    # If both in bomb set, can play the bigger bomb
                    elif float(pattern) > float(prev_pattern):
                        curr_mask[i] = True
                        
                    # Else, accept/reject action normally
                    
        # Able to skip if not free, unable to skip if free
        curr_mask[-1] = not free
        
    return hand_mask, curr_mask


def calculate_reward(pattern, num_cards_played, remainder):
    """
    Calculate the reward for a particular action.
    Note: Losses are calculated only after the game ends.
    """
    if remainder == 0:
        return REWARDS["win"]
    elif pattern == "skip":
        return REWARDS["skip"]
    elif pattern in ["claim_landlord", "refuse_landlord"]:
        return 0
    else:
        return REWARDS["valid"] * num_cards_played
    
    
# Convert frequency array to card string
def freq_array_to_card_str(hand):
    card_str = ""
    for rank, freq in enumerate(hand):
        card_str += CARDS[rank] * freq + " "
    return card_str


def read_user_cards(pattern, user_cards, available_actions):
    """
    Convert card string to frequency array.
    """
    choice = [0] * 15
    leading_rank = RANKS[user_cards[0]]
    
    # Note: Already made sure user can't both be free and skip turn
    # So if the input is empty at this point, available_actions[-1] must be "skip"
    if not user_cards:
        return available_actions[-1]
    
    for c in user_cards:
        choice[RANKS[c]] += 1
    
    # Check if the specific choice is playable
    if (pattern, leading_rank, choice) in available_actions:
        return pattern, leading_rank, choice, True
    else:
        return None, None, None, False
    
    
# Print the current state of the game
def print_game(pattern, leading_rank, choice, new_state, players, curr_player, start=False, verbose=False):
    if pattern == "skip":
        print(f"Skip. Skip count: {new_state["curr_skips"]}\n")
        print(f"Next player: {new_state["opponents"]["id"][0]}\n")
    else:
        if start:
            if verbose:
                for idx, player in enumerate(players):
                    print(f"Player {idx}: {player.hand} {sum(player.hand)} {type(player).__name__} ({'Landlord' if player.landlord else 'Peasant'})")
            else:
                for idx, player in enumerate(players):
                    print(f"Player {idx}: {sum(player.hand)} remaining ({'Landlord' if player.landlord else 'Peasant'})")
            print(f"\nPlayer {curr_player} starts\n")

        else:
            print(f"Choice: [{freq_array_to_card_str(choice)}], pattern: {pattern}, rank: {leading_rank}\n")
            if verbose:
                for idx, player in enumerate(players):
                    print(f"Player {idx}: {player.hand} {sum(player.hand)} {type(player).__name__} ({'Landlord' if player.landlord else 'Peasant'})")
            else:
                for idx, player in enumerate(players):
                    print(f"Player {idx}: {sum(player.hand)} remaining ({'Landlord' if player.landlord else 'Peasant'})")
                    
            print(f"All remaining: {freq_array_to_card_str(new_state["opponents"]["all_cards_remaining"])}\n")
            print(f"Next player: {new_state["opponents"]["id"][0]}\n")

    
def announce_winner(mode, curr_player, winner_is_landlord):
    if mode == "indv":
        print(f"Game over. Player {curr_player} wins!")
    else:
        if winner_is_landlord:
            print("Game over. Landlord wins!")
        else:
            print("Game over. Peasants win!")
