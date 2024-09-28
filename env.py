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
    
    
    def step(self, players, curr_player, pattern, choice, leading_rank, remainder, landlord_cards):
        """
        Execute a single step in the game.
        The current player makes a move, and the environment returns the new state, reward, and done flag.
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
        action = {
            "player": curr_player,
            "pattern": pattern,
            "choice": choice,
            "leading_rank": leading_rank
        }
        self.action_history.append(action)
        new_state = self.get_state(players, curr_player)
        reward = utils.calculate_reward(pattern, sum(choice), remainder)

        # Check if the game is over
        done = remainder <= 0
        
        return action, new_state, reward, done
    
    
    def get_state(self, players, curr_player):
        """
        Record the current state of the game.
        """
        player_id = curr_player
        opponent_ids = [(curr_player + i) % self.num_players for i in range(1, self.num_players)]
        
        state = {"self": {"id":           player_id,
                          "free":         players[player_id].free,
                          "is_landlord":  players[player_id].landlord,
                          "hand":         players[player_id].hand.tolist(),
                          "num_remaining":int(self.num_remaining[player_id]),
                          "cards_played": self.cards_played[player_id].tolist(),
                          "bombs_played": int(self.bombs_played[player_id]),
                          "bomb_types":   list(self.bomb_types_played[player_id]),
                          "total_skips":  int(self.total_skips[player_id])},
                    
                  "opponents": {"id":                    opponent_ids,
                                "is_landlord":           [players[p].landlord for p in opponent_ids],
                                "num_remaining":         self.num_remaining[opponent_ids].tolist(),
                                "each_opp_cards_played": self.cards_played[opponent_ids].tolist(),
                                "opp_cards_remaining":   (self.cards_remaining - players[player_id].hand).tolist(),
                                "all_cards_remaining":   self.cards_remaining.tolist(),
                                "bombs_played":          self.bombs_played[opponent_ids].tolist(),
                                "bomb_types":            [list(self.bomb_types_played[p]) for p in opponent_ids],
                                "total_skips":           self.total_skips[opponent_ids].tolist()},
            
                   "curr_skips":        self.curr_skips,
                   "mode":              self.mode,
                   "choosing_landlord": self.choosing_landlord,
                   "refused_landlord":  self.refused_landlord,
                   "action_history":    self.action_history.copy()}
        
        return state