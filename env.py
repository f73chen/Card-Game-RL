import random
import numpy as np
import utils

from consts import *

class GameEnv:
    def __init__(self, num_decks=1, num_players=4, mode="lord", moveset=MOVESET_1):
        # Variables that persist across multiple games
        self.num_decks = num_decks
        self.num_players = num_players
        self.mode = mode
        
        all_moves = utils.get_all_moves()
        cards_remaining = np.array(CARD_FREQ) * self.num_decks
        self.deck_moves = utils.get_deck_moves(all_moves, cards_remaining, moveset)
    
        
    def reset(self, players):
        # Variables that only persist for 1 game
        self.action_history = []
        
        self.num_remaining = np.array([sum(player.hand) for player in players])
        self.cards_played = np.zeros((self.num_players, NUM_RANKS))
        self.cards_remaining = np.array(CARD_FREQ) * self.num_decks
        self.bombs_played = np.zeros(self.num_players).astype(int)
        self.bomb_types_played = [set() for _ in range(self.num_players)]
        self.total_skips = np.zeros(self.num_players).astype(int)
        self.landlord_idx = None
        self.choosing_landlord = self.mode=="lord"
        self.refused_landlord = []
        
        curr_player = 0
        initial_state = utils.get_state(self.num_players, players, curr_player, self.action_history, 
                                        self.num_remaining, self.cards_played, self.cards_remaining, self.bombs_played, self.bomb_types_played, self.total_skips, self.curr_skips,
                                        self.mode, self.choosing_landlord, self.refused_landlord)
        return initial_state
    
    
    def step(self, players, curr_player, pattern, choice, leading_rank, remainder, landlord_cards):
        """
        Execute a single step in the game.
        The current player makes a move, and the environment returns the new state, reward, and done flag.
        
        params:
            action (dict): A dictionary containing move information (e.g., pattern, choice)
        """
        # Initial setup: claiming the landlord cards
        if pattern in ["claim_landlord", "refuse_landlord"]:
            if pattern == "claim_landlord":
                self.landlord_idx = curr_player
            else:
                self.refused_landlord.append(curr_player)
            
            # If all players refused, player 0 automatically becomes the landlord
            if len(self.refused_landlord) == self.num_players:
                self.landlord_idx = 0
            
            if self.landlord_idx is not None:
                players[self.landlord_idx].landlord = True
                players[self.landlord_idx].hand += landlord_cards
                self.choosing_landlord = False
            
            self.num_remaining[self.landlord_idx] = sum(players[self.landlord_idx].hand)

        # Normal play
        else:
            skipped = pattern == "skip"
            if skipped:
                self.total_skips[curr_player] += 1  # Total number of skips by the current player
                self.curr_skips += 1                # Increment the temporary skip count

            else:
                self.cards_played[curr_player] += choice    # Frequency of cards played by the current player
                self.cards_remaining -= choice              # Frequency of cards remaining in play
                self.curr_skips = 0                         # Reset the temporary skip count

                if pattern in BOMB_SET:
                    self.bombs_played[curr_player] += 1                # Number of bombs played by the current player
                    self.bomb_types_played[curr_player].add(pattern)   # Types of bombs played by the current player
            
            self.num_remaining[curr_player] = remainder    # Update the number of cards remaining in the player's hand


        # Record the action and new state
        action_record = {
            "player": curr_player,
            "pattern": pattern,
            "choice": choice,
            "leading_rank": leading_rank
        }
        self.action_history.append(action_record)
        new_state = utils.get_state(self.num_players, players, curr_player, self.action_history, 
                                    self.num_remaining, self.cards_played, self.cards_remaining, self.bombs_played, self.bomb_types_played, self.total_skips, self.curr_skips,
                                    self.mode, self.choosing_landlord, self.refused_landlord)
        reward = utils.calculate_reward(skipped, sum(choice), remainder)

        # Check if the game is over
        done = remainder <= 0
        
        return new_state, reward, done