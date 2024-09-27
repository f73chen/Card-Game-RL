import random
import numpy as np
import utils

from consts import *

class GameEnv:
    def __init__(self, num_decks=1, num_players=4, mode="indv", players=[], moveset=MOVESET_1, seed=None):
        """
        Initialize variables that are constant across all games.
        """
        # Variables that persist across multiple games
        # self.num_decks = num_decks
        # self.num_players = num_players
        # self.mode = mode
        # self.moveset = moveset
        
        # all_moves = utils.get_all_moves()
        # cards_remaining = np.array(CARD_FREQ) * self.num_decks
        # self.deck_moves = utils.get_deck_moves(all_moves, cards_remaining, moveset)
        
        # if seed is not None:
        #     random.seed(seed)
        
        
    def reset(self, players):
        """
        Initialize a fresh game instance.
        """
        # Variables that only persist for 1 game
        # self.game_history = []
        # self.action_history = []
        # self.landlord_idx = None
        # self.curr_player = 0
        
        # num_remaining = np.array([sum(player.hand) for player in players])
        # self.cards_played = np.zeros((self.num_players, NUM_RANKS))
        # self.cards_remaining = np.array(CARD_FREQ) * self.num_decks
        # self.bombs_played = np.zeros(self.num_players).astype(int)
        # self.bomb_types_played = [set() for _ in range(self.num_players)]
        # self.total_skips = np.zeros(self.num_players).astype(int)
        
        # self.curr_skips = 0
        # self.pattern = None
        # self.prev_choice = None
        # self.leading_rank = None
        
    def step(self, players, curr_player, valid_move, pattern, choice, leading_rank, remainder):
        """
        Execute a single step in the game.
        The current player makes a move, and the environment returns the new state, reward, and done flag.
        
        params:
            action (dict): A dictionary containing move information (e.g., pattern, choice)
        """

        # Update information based on if the move was valid
        if valid_move:
            self.cards_played[curr_player] += choice  # Frequency of cards played by the current player
            self.cards_remaining -= choice                 # Frequency of cards remaining in play
            self.curr_skips = 0                                 # Reset the temporary skip count

            if pattern in BOMB_SET:
                self.bombs_played[curr_player] += 1                # Number of bombs played by the current player
                self.bomb_types_played[curr_player].add(pattern)   # Types of bombs played by the current player
        
        else:
            self.total_skips[curr_player] += 1                 # Total number of skips by the current player
            self.curr_skips += 1                                    # Increment the temporary skip count

        self.num_remaining[curr_player] = remainder    # Update the number of cards remaining in the player's hand

        # Record the action and new state
        action_record = {
            "player": curr_player,
            "valid_move": valid_move,
            "pattern": pattern,
            "choice": choice,
            "leading_rank": leading_rank
        }
        self.action_history.append(action_record)
        new_state = utils.get_state(self.num_players, players, curr_player, self.action_history, self.num_remaining, self.cards_played, self.cards_remaining, self.bombs_played, self.bomb_types_played, self.total_skips, self.curr_skips)
        reward = utils.calculate_reward(valid_move, sum(choice), remainder)

        # Check if the game is over
        done = remainder <= 0
        
        return new_state, reward, done