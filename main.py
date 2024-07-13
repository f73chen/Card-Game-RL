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
            
        # Decide the order that cards were dealt
        self.order = random.randint(0, self.num_players-1)
        self.hands = self.hands[self.order:] + self.hands[:self.order]
        self.players = [NaivePlayer(hand=self.hands[p], moveset=self.moveset) for p in range(self.num_players)]
        
    def play_game(self):
        for player in self.players:
            print(player.hand)
        print()
        
        # Start with a random player
        curr_player = random.randint(0, self.num_players-1)
        print(f"Player {curr_player} starts")
        
        self.players[curr_player].free = True
        pattern = None
        prev_choice = None
        leading_rank = None
        skip_count = 0
        
        # Players play in order until the game is over (empty hand)
        while True:
            # If all other players skip their turn, the current player is free to move
            if skip_count == self.num_players - 1:
                self.players[curr_player].free = True
                skip_count = 0
            
            contains_pattern, pattern, prev_choice, leading_rank, remainder = self.players[curr_player].move(pattern=pattern, prev_choice=prev_choice, leading_rank=leading_rank)
            
            if not contains_pattern:
                skip_count += 1
            else:
                skip_count = 0
            
            print(f"Player {curr_player} plays:")
            if contains_pattern:
                print(prev_choice, pattern, leading_rank)
                print(self.players[curr_player].hand, remainder)
                print()
                for player in self.players:
                    print(player.hand)
            else:
                print(f"Skip. Skip count: {skip_count}")
            print()
        
            # Check if the game is over
            if remainder <= 0:
                break
            
            # Move on to the next player
            curr_player = (curr_player + 1) % self.num_players
            
        print(f"Game over. Winner is player {curr_player}")

class NaivePlayer:
    def __init__(self, hand, moveset, free=False):
        self.hand = hand
        self.moveset = moveset
        self.free = free
        
    def move(self, pattern=None, prev_choice=None, leading_rank=-1):
        # If free to move, play the smallest available hand
        # Doesn't care about previous cards or leading rank
        if self.free:
            self.free = False
            random.shuffle(self.moveset)
            for pattern in self.moveset:
                contains_pattern, choice, leading_rank = utils.smallest_valid_choice(hand=self.hand, pattern=pattern)
                if contains_pattern:
                    break
        
        # Else follow the pattern of the player before it and play a higher rank
        else:
            contains_pattern, choice, leading_rank = utils.smallest_valid_choice(hand=self.hand, pattern=pattern, leading_rank=leading_rank)
                
        # Return the card choice and subtract it from its hand
        choice = np.array(choice)
        self.hand -= choice
        return contains_pattern, pattern, choice, leading_rank, np.sum(self.hand)
            
env = GameEnv(num_decks=2)
env.play_game()

