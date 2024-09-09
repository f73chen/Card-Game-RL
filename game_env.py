import random
import numpy as np
import utils
from consts import *
from players import DefaultPlayer, UserPlayer, RLPlayer


class GameEnv:
    def __init__(self, num_decks=1, num_players=4, players=[], mode="indv", moveset=MOVESET_1, seed=None):
        """
        Initialize variables that are constant across all games.
        
        params:
            num_decks   (int):      Number of decks of cards to use (1 or 2)
            num_players (int):      Number of players in the game (3 or 4)
            players     (Player):   List of player objects
            mode        (str):      "indv" for individual or "lord" for landlord
            moveset     (array):    Simplified or comprehensive moveset
            seed        (int):      Random seed
        """
        self.num_players = num_players
        self.num_decks = num_decks
        self.players = players
        self.mode = mode
        self.moveset = moveset
        self.game_history = []      # Record the game history for training
        self.action_history = []    # Record the action history for training
        self.all_remaining = None   # Frequency of all cards left in play
        self.landlord_idx = None    # Index of the landlord player
        self.curr_player = 0        # Index of the current player
        
        if seed is not None:
            random.seed(seed)
        
        # Adjust the number of players and decks to be consistent with the win mode
        self.adjust_player_count()
        
        
    def adjust_player_count(self):
        """
        Adjust the number of players and decks based on the win mode.
        """
        # Individual can have 3 or 4 players, landlord can only have 3 players
        if self.mode == "indv":
            if self.num_players < 3:
                self.num_players = 3
            elif self.num_players > 4:
                self.num_players = 4
        elif self.mode == "lord":
            self.num_players = 3
        
        # Number of decks can only be 1 or 2
        if self.num_decks < 1:
            self.num_decks = 1
        elif self.num_decks > 2:
            self.num_decks = 2
        
        # Remove extra players and add default players if necessary
        self.players = self.players[:self.num_players]
        for p in range(self.num_players - len(self.players)):
            self.players.append(DefaultPlayer())
            
        # Apply the same moveset to all players
        for player in self.players:
            player.moveset = self.moveset
            
        # Shuffle player order
        random.shuffle(self.players)
        
        
    def deal_cards(self, num_players, num_decks, mode):
        """
        Deal cards to the players based on the number of players and win mode.
        
        params:
            num_players (int):  Number of players in the game
            num_decks   (int):  Number of decks of cards to use
            mode        (str):  "indv" for individual or "lord" for landlord
        """
        card_freq = np.array(CARD_FREQ) * num_decks # Total number of cards in the deck(s)
        
        if mode == "indv":
            cards_per_player = CARDS_PER_PLAYER["indv"][num_players][num_decks]
            hands = [np.array([0] * NUM_RANKS) for _ in range(num_players)] # Start with empty hands
            
            # Deal cards for each player
            for p in range(num_players-1):
                for card in range(cards_per_player[p]):
                    dealt = False
                    while not dealt:
                        idx = random.randint(0, NUM_RANKS-1)
                        if card_freq[idx] > 0:
                            card_freq[idx] -= 1
                            hands[p][idx] += 1
                            dealt = True
            
            # Deal the remaining cards to the last player
            hands[-1] = card_freq
            
            # Assign the hands to the player objects
            for p in range(num_players):
                self.players[p].hand = hands[p]
            
        elif mode == "lord":
            cards_per_player = CARDS_PER_PLAYER["lord"][num_players][num_decks][0]
            hands = [np.array([0] * NUM_RANKS) for _ in range(num_players)]
            
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
            
            # Assign the hands to the player objects
            # The remaining cards are the landlord cards
            for p in range(num_players):
                self.players[p].hand = hands[p]
                
            # Players take turns to claim the landlord cards, starting from player 0
            for p in range(num_players):
                claimed = self.players[p].claim_landlord(card_freq)
                print(f"Player {p} claims the landlord cards: {claimed}\n")
                if claimed:
                    self.landlord_idx = p
                    self.curr_player = p    # The landlord always starts first
                    break
                
            # If all refuse, player 0 must become the landlord
            if self.landlord_idx is None:
                self.landlord_idx = 0
                self.players[0].landlord = True
                
            # Add the cards to the landlord's hand
            self.players[self.landlord_idx].hand += card_freq


    def reset(self):
        """
        Initialize a fresh game instance.
        """
        # Reset the game history
        self.game_history = []
        self.action_history = []
        self.landlord_idx = None
        self.curr_player = 0
        self.all_remaining = np.array(CARD_FREQ) * self.num_decks    # Frequency of all cards left in play
            
        print(f"New {self.mode} game with {self.num_players} players and {self.num_decks} deck(s) of cards.")
        
        # Deal the regular hand, then the landlord hand
        print("Dealing cards...")
        self.deal_cards(self.num_players, self.num_decks, self.mode)
        
        
    def play_game(self, verbose=False):
        """
        Lets players move in order until the game is over.
        """
        # Reset temporary variables
        self.players[self.curr_player].free = True
        valid_move = True
        pattern = None
        prev_choice = None
        leading_rank = None
        remainder = sum(self.players[self.curr_player].hand)
        skip_count = 0
                
        utils.print_game(valid_move, pattern, prev_choice, leading_rank, self.all_remaining, skip_count, self.curr_player, self.players, start=True, verbose=False)
        
        # Players play in order until the game is over (empty hand)
        while True:
            # Record the current state
            curr_state = self.get_state()
            
            # If all other players skip their turn, the current player is free to move
            if skip_count == self.num_players - 1:
                self.players[self.curr_player].free = True
                skip_count = 0
            
            # Player makes a move
            valid_move, pattern, prev_choice, leading_rank, remainder = self.players[self.curr_player].move(pattern=pattern, prev_choice=prev_choice, leading_rank=leading_rank)
            
            # Update the remaining cards in play
            if valid_move:
                self.all_remaining -= prev_choice
                skip_count = 0
            # If the player didn't make a move, increment the skip count
            else:
                skip_count += 1
            
            # Record the player action and new state
            action = {"player":         self.curr_player,
                      "valid_move":     valid_move,
                      "pattern":        pattern,
                      "choice":         prev_choice.tolist(),
                      "leading_rank":   leading_rank}
            self.action_history.append(action)
            new_state = self.get_state()    # New state's action history includes the current action
            reward = self.calculate_reward(valid_move, remainder)
            self.game_history.append({"state":      curr_state, 
                                     "action":      action, 
                                     "new_state":   new_state, 
                                     "reward":      reward})
            
            # Check if the game is over
            if remainder <= 0:
                break
            
            # Else, continue to the next player
            self.curr_player = (self.curr_player + 1) % self.num_players
            
            utils.print_game(valid_move, pattern, prev_choice, leading_rank, self.all_remaining, skip_count, self.curr_player, self.players, start=False, verbose=False)
            
        # TODO: Record the game over in game history and update reward for all players based on game outcome
        if self.mode == "indv":
            print(f"Game over. Player {self.curr_player} wins!")
        else:
            print(f"Game over. {'Landlord' if self.curr_player == self.landlord_idx else 'Peasants'} win!")
        return self.game_history
        
    
    def get_state(self):
        """
        Record the current state of the game.
        """
        state = {
            "curr_player":      self.curr_player,
            "self_free":        self.players[self.curr_player].free,
            "is_landlord":      self.players[self.curr_player].landlord,
            "hand":             self.players[self.curr_player].hand.tolist(),
            "num_remaining":    int(sum(self.players[self.curr_player].hand)),
            "all_remaining":    self.all_remaining.tolist(),    # Convert to list for JSON
            "opp_remaining":    (self.all_remaining - self.players[self.curr_player].hand).tolist(),
            "action_history":   self.action_history.copy()
        }
        return state


    def calculate_reward(self, valid_move, remainder):
        # TODO: Improve reward calculation
        """
        Calculate the reward for a particular action.
        
        params:
            valid_move (bool):    Whether the player made a valid move
            remainder  (int):     Number of cards remaining in the player's hand
        """
        if remainder == 0:
            return 10   # Win
        elif valid_move:
            return 0.1  # Successful move
        else:
            return -0.1 # Skipped turn
        
        
    def replay(self, history):
        """
        Re-enact the game based on the recorded history.
        
        params:
            history (list): List of dictionaries containing state, action, new state, and reward
        """
        # TODO: Change how the history is replayed
        for step in history:
            print(f"Player {step['action']['player']} action:")
            print(f"State: {step['state']}")
            print(f"Action: {step['action']}")
            print(f"New State: {step['new_state']}")
            print(f"Reward: {step['reward']}")
            print()
            
            
    # TODO: Implement playing with RL model
    # def play_with_model(self, model, num_games=1):
    #     for _ in range(num_games):
    #         state = self.reset()
    #         done = False
    #         while not done:
    #             with torch.no_grad():
    #                 action = torch.argmax(model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))).item()
    #             state, reward, done, _ = self.step(action)
    #             # Add logic to handle NPCs and human players
    #             # For NPCs, you can use predefined strategies or random actions
    #             # For human players, you can prompt for input
 