import numpy as np
import random

import utils
from consts import *

class GameEnv:
    def __init__(self, num_players=4, num_decks=1, mode="individual", moveset=MOVESET_1):
        self.num_players = num_players
        self.num_decks = num_decks
        self.mode = mode
        self.moveset = moveset
        
        self.reset()

    def reset(self):
        print(f"New game with {self.num_players} players and {self.num_decks} deck(s) of cards.")
        
        card_freq = np.array(CARD_FREQ) * self.num_decks
        cards_per_player = CARDS_PER_PLAYER[f"{self.num_players}_{self.num_decks}"]
        
        # Randomly divide into 4 piles of cards then decide which player gets which pile and who starts
        # Game mode must be individual or pairs (no landlord; assume individual for now)
        if self.num_players == 4:
            # Players start with empty hands
            self.hands = [np.array([0] * NUM_RANKS) for _ in range(self.num_players)]
            
            # Deal cards for each player
            for p in range(self.num_players-1):
                for card in range(cards_per_player[p]):
                    dealt = False
                    while not dealt:
                        idx = random.randint(0, NUM_RANKS-1)
                        if card_freq[idx] > 0:
                            card_freq[idx] -= 1
                            self.hands[p][idx] += 1
                            dealt = True
                            
            self.hands[-1] = card_freq
            
        self.order = random.randint(0, self.num_players-1)
        self.hands = self.hands[self.order:] + self.hands[:self.order]
        self.players = [NaivePlayer(hand=self.hands[p], moveset=self.moveset) for p in range(self.num_players)]
        
        self.play_game()
        
    def play_game(self):
        curr_player = random.randint(0, self.num_players-1)
        self.players[curr_player].free = True
        
        counter = 0
        # while True:
        while counter <= 10:
            pattern, choice, remainder = self.players[curr_player].move(prev_pattern=pattern, prev_choice=choice)
            curr_player = (curr_player + 1) % self.num_players
            
            print(self.players[curr_player].hand, pattern, choice)
            
            if remainder <= 0:
                break
            counter += 1
            
        print(f"Game over. Winner is player {curr_player}")

class NaivePlayer:
    def __init__(self, hand, moveset, free=False):
        self.hand = hand
        self.moveset = moveset
        self.free = free
        
    def move(self, prev_pattern="", prev_choice=""):
        if self.free:
            for pattern in self.moveset:
                if utils.valid_pattern(hand=self.hand, pattern=pattern):
                    prev_pattern = pattern
                    choice = utils.smallest_valid_play(hand=self.hand, pattern=pattern)
                    break
        else:
            if utils.valid_pattern(hand=self.hand, pattern=prev_pattern):
                choice = utils.smallest_valid_play(hand=self.hand, pattern=pattern, cards=prev_choice)
                
        self.hand -= choice
        return prev_pattern, choice, np.sum(self.hand)
            
env = GameEnv()

