import random
import numpy as np
import utils
from consts import *
from players import DefaultPlayer, UserPlayer, RLPlayer


class GameEnv:
    # Overwritten
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
        self.landlord_idx = None    # Index of the landlord player
        self.curr_player = 0        # Index of the current player
        
        if seed is not None:
            random.seed(seed)
        
        # Adjust the number of players and decks to be consistent with the win mode
        self.adjust_player_count()
        
    # Will be inherited
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
           
    # Will be inherited  
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

    # Overwritten
    def reset(self):
        """
        Initialize a fresh game instance.
        """
        # Reset the game history
        self.game_history = []
        self.action_history = []
        self.landlord_idx = None
        self.curr_player = 0
            
        print(f"New {self.mode} game with {self.num_players} players and {self.num_decks} deck(s) of cards.")
        
        # Deal the regular hand, then the landlord hand
        print("Dealing cards...")
        self.deal_cards(self.num_players, self.num_decks, self.mode)

    # Adapted into .step(action)
    def play_game(self, verbose=False):
        """
        Lets players move in order until the game is over.
        """
        # Reset temporary variables
        self.players[self.curr_player].free = True  # TODO !!!
        valid_move = True                           # TODO !!!
        
        curr_skips = 0
        pattern = None
        prev_choice = None
        leading_rank = None
        
        # Track public information for state recording
        num_remaining = np.array([sum(player.hand) for player in self.players])
        cards_played = np.zeros((self.num_players, NUM_RANKS))
        cards_remaining = np.array(CARD_FREQ) * self.num_decks
        bombs_played = np.zeros(self.num_players).astype(int)
        bomb_types_played = [set() for _ in range(self.num_players)]
        total_skips = np.zeros(self.num_players).astype(int)
                
        utils.print_game(valid_move, pattern, prev_choice, leading_rank, cards_remaining, curr_skips, self.curr_player, self.players, start=True, verbose=False)
        
        # Players play in order until the game is over (empty hand)
        while True:
            # Record the current state
            curr_state = self.get_state(num_remaining, cards_played, cards_remaining, bombs_played, bomb_types_played, total_skips, curr_skips)
            
            # If all other players skip their turn, the current player is free to move
            if curr_skips == self.num_players - 1:
                self.players[self.curr_player].free = True
                curr_skips = 0
            
            # Player makes a move
            valid_move, pattern, prev_choice, leading_rank, remainder = self.players[self.curr_player].move(pattern=pattern, prev_choice=prev_choice, leading_rank=leading_rank)
            num_remaining[self.curr_player] = remainder   # Update the number of cards remaining in the player's hand
            
            # Update the remaining cards in play
            if valid_move:
                cards_played[self.curr_player] += prev_choice
                cards_remaining -= prev_choice
                curr_skips = 0
                
                if pattern in BOMB_SET:
                    bombs_played[self.curr_player] += 1
                    bomb_types_played[self.curr_player].add(pattern)
                    
            # If the player didn't make a move, increment the skip count
            else:
                total_skips[self.curr_player] += 1
                curr_skips += 1
            
            # Record the player action and new state
            action_record = {
                "player":         self.curr_player,
                "valid_move":     valid_move,
                "pattern":        pattern,
                "choice":         prev_choice.tolist(),
                "leading_rank":   leading_rank
            }
            self.action_history.append(action_record)
            new_state = self.get_state(num_remaining, cards_played, cards_remaining, bombs_played, bomb_types_played, total_skips, curr_skips)    # Note: New state's action history includes the current action
            reward = self.calculate_reward(valid_move, sum(prev_choice), remainder)
            
            self.game_history.append({
                "state": curr_state, 
                "action": action_record, 
                "new_state": new_state, 
                "reward": reward,  # Immediate reward only
                "done": False
            })
            
            next_player = (self.curr_player + 1) % self.num_players
            utils.print_game(valid_move, pattern, prev_choice, leading_rank, cards_remaining, curr_skips, next_player, self.players, start=False, verbose=False)
            
            # Check if the game is over
            if remainder <= 0:
                break
            
            # Else, continue to the next player
            self.curr_player = next_player
            
        # Announce the game result and update rewards for all players
        if self.mode == "indv":
            print(f"Game over. Player {self.curr_player} wins!")
            self.finalize_rewards(winners=[self.curr_player])
        else:
            landlord_wins = self.curr_player == self.landlord_idx
            if landlord_wins:
                print("Game over. Landlord wins!")
                self.finalize_rewards(winners=[self.landlord_idx])
            else:
                print("Game over. Peasants win!")
                self.finalize_rewards(winners=[p for p in range(self.num_players) if p != self.landlord_idx])
        return self.game_history
    
    # Overwritten
    def finalize_rewards(self, winners):
        """
        Update player rewards after the game ends.
        For example, in a 3-player game, the last 3 records in the game history are updated.
        Landlord rewards are multiplied by the number of peasants.
        
        params:
            winners (list): List of player indices who won the game
        """
        if self.mode == "indv":
            for p in range(self.num_players):
                updated_entry = self.game_history[-p-1]
                curr_player = updated_entry["state"]["self"]["id"]
                if curr_player in winners:
                    updated_entry["reward"] = REWARDS["win"]
                else:
                    updated_entry["reward"] = REWARDS["loss"]
                updated_entry["done"] = True
                self.game_history[-p-1] = updated_entry
        else:
            for p in range(self.num_players):
                updated_entry = self.game_history[-p-1]
                curr_player = updated_entry["state"]["self"]["id"]
                if curr_player in winners:
                    updated_entry["reward"] = REWARDS["win"] * (self.num_players - len(winners))
                else:
                    updated_entry["reward"] = REWARDS["loss"] * len(winners)
                updated_entry["done"] = True
                self.game_history[-p-1] = updated_entry
        
    # Will be inherited        
    def get_state(self, num_remaining, cards_played, cards_remaining, bombs_played, bomb_types_played, total_skips, curr_skips):
        """
        Record the current state of the game.
        """
        player_id = self.curr_player
        opponent_ids = [(self.curr_player + i) % self.num_players for i in range(1, self.num_players)]
        
        state = {"self": {"id":           player_id,
                          "free":         self.players[player_id].free,
                          "is_landlord":  self.players[player_id].landlord,
                          "hand":         self.players[player_id].hand.tolist(),
                          "num_remaining":int(num_remaining[player_id]),
                          "cards_played": cards_played[player_id].tolist(),
                          "bombs_played": int(bombs_played[player_id]),
                          "bomb_types":   list(bomb_types_played[player_id]),
                          "total_skips":  int(total_skips[player_id])},
                 
                 "opponents": {"id":                    opponent_ids,
                               "is_landlord":           [self.players[p].landlord for p in opponent_ids],
                               "num_remaining":         num_remaining[opponent_ids].tolist(),
                               "each_opp_cards_played": cards_played[opponent_ids].tolist(),
                               "opp_cards_remaining":   (cards_remaining - self.players[player_id].hand).tolist(),
                               "all_cards_remaining":   cards_remaining.tolist(),
                               "bombs_played":          bombs_played[opponent_ids].tolist(),
                               "bomb_types":            [list(bomb_types_played[p]) for p in opponent_ids],
                               "total_skips":           total_skips[opponent_ids].tolist()},
            
                 "curr_skips": curr_skips,
                 "action_history":   self.action_history.copy()}
        
        return state

    # Will be inherited
    def calculate_reward(self, valid_move, num_cards_played, remainder):
        """
        Calculate the reward for a particular action.
        Note: Losses are calculated only after the game ends.
        
        params:
            valid_move (bool):    Whether the player made a valid move
            remainder  (int):     Number of cards remaining in the player's hand
        """
        if remainder == 0:
            return REWARDS["win"]
        elif valid_move:
            return REWARDS["valid"] * num_cards_played
        else:
            return REWARDS["pass"]
        
    # Not passed on
    def replay(self, history):
        """
        Re-enact the game based on the recorded history.
        
        params:
            history (list): List of dictionaries containing state, action, new state, and reward
        """
        for step in history:
            print(f"Player {step['action']['player']} makes a move.")
            print(f"Pattern: {step['action']['pattern']}")
            print(f"Choice: {utils.freq_array_to_card_str(step['action']['choice'])}")
            print(f"Remaining cards in hand: {step['new_state']['num_remaining']}")
            print()


class TrainGameEnv(GameEnv):
    def __init__(self, num_decks=1, num_players=4, players=[], mode="indv", moveset=MOVESET_1, seed=None):
        # Initialize variables that are constant across all games
        # Also adjust player count to be consistent with the win mode
        super().__init__(num_decks, num_players, players, mode, moveset, seed)

        # Track public information for state recording
        self.num_remaining = None
        self.cards_played = None
        self.cards_remaining = None
        self.bombs_played = None
        self.bomb_types_played = None
        self.total_skips = None
        self.curr_skips = 0
        self.pattern = None
        self.prev_choice = None
        self.leading_rank = None
        
    def reset(self):
        """
        Initialize a fresh game instance.
        """
        # Reset the game history
        self.game_history = []
        self.action_history = []
        self.landlord_idx = None
        self.curr_player = 0

        # Reset game variables
        self.num_remaining = np.array([0] * self.num_players)   # 0 for now
        self.cards_played = np.zeros((self.num_players, NUM_RANKS))
        self.cards_remaining = np.array(CARD_FREQ) * self.num_decks
        self.bombs_played = np.zeros(self.num_players).astype(int)
        self.bomb_types_played = [set() for _ in range(self.num_players)]
        self.total_skips = np.zeros(self.num_players).astype(int)
        
        self.curr_skips = 0
        self.pattern = None
        self.prev_choice = None
        self.leading_rank = None

        # Deal the regular hand, then the landlord hand
        # TODO
        self.deal_cards(self.num_players, self.num_decks, self.mode)

        # Update num_remaining after dealing the cards
        self.num_remaining = np.array([sum(player.hand) for player in self.players])
        
        # After dealing cards, self.curr_player is the player to start
        # Set the first player to be free to move
        self.players[self.curr_player].free = True

        return self.get_state(self.num_remaining, self.cards_played, self.cards_remaining, self.bombs_played, self.bomb_types_played, self.total_skips, self.curr_skips)

    def step(self, action):
        """
        Execute a single step in the game.
        The current player makes a move, and the environment returns the new state, reward, and done flag.
        
        params:
            action (dict): A dictionary containing move information (e.g., pattern, choice)
        """
        
        # Extract action details
        # Note: remainder is not given but calculated after the fact
        valid_move = action.get("valid_move")
        pattern = action.get("pattern")
        prev_choice = action.get("choice")
        leading_rank = action.get("leading_rank")

        # Update information based on if the move was valid
        if valid_move:
            self.cards_played[self.curr_player] += prev_choice  # Frequency of cards played by the current player
            self.cards_remaining -= prev_choice                 # Frequency of cards remaining in play
            self.curr_skips = 0                                 # Reset the temporary skip count
            remainder = sum(self.players[self.curr_player].hand) - sum(prev_choice) # Total number of cards remaining in the current player's hand
        
            if pattern in BOMB_SET:
                self.bombs_played[self.curr_player] += 1                # Number of bombs played by the current player
                self.bomb_types_played[self.curr_player].add(pattern)   # Types of bombs played by the current player
        
        else:
            remainder = sum(self.players[self.curr_player].hand)    # Total number of cards remaining in the current player's hand
            self.total_skips[self.curr_player] += 1                       # Total number of skips by the current player
            self.curr_skips += 1                                    # Increment the temporary skip count

        self.num_remaining[self.curr_player] = remainder    # Update the number of cards remaining in the player's hand

        # Record the action and new state
        action_record = {
            "player": self.curr_player,
            "valid_move": valid_move,
            "pattern": pattern,
            "choice": prev_choice,
            "leading_rank": leading_rank
        }
        self.action_history.append(action_record)
        new_state = self.get_state(self.num_remaining, self.cards_played, self.cards_remaining, self.bombs_played, self.bomb_types_played, self.total_skips, self.curr_skips)
        reward = self.calculate_reward(valid_move, sum(prev_choice), remainder)

        # Check if the game is over
        done = remainder <= 0

        # Continue to the next player
        self.curr_player = (self.curr_player + 1) % self.num_players

        return new_state, reward, done, {}


