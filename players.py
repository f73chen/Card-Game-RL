import numpy as np
import random
import utils
from consts import *

            
class Player:
    def __init__(self):
        self.hand = None
        self.moveset = None
        self.landlord = False
        self.free = False
                   
            
class DefaultPlayer(Player):
    def __init__(self):
        super().__init__()
        
        
    def select_action(self, state):
        """
        Make a random valid move based on the current state of the game.
        
        params:
            state (dict): The current state of the game
            
        returns:
            valid_move   (bool): Whether the player made a valid move
            pattern      (str):  The pattern of the move
            choice       (array): The cards played by the player
            leading_rank (int):  The rank of the leading card
            remainder    (int):  Number of cards remaining in the player's hand
        """
        if not state["action_history"]:
            pattern = None
            leading_rank = -1
        else:       
            pattern = state["action_history"][-1]["pattern"]
            leading_rank = state["action_history"][-1]["leading_rank"]
        
        # If free to move, play the smallest available hand for a random available pattern
        if self.free:
            leading_rank = -1   # Reset the leading rank
            self.free = False
            random.shuffle(self.moveset)
            for pattern in self.moveset:
                valid_move, pattern, choice, leading_rank = utils.smallest_valid_choice(hand=self.hand, pattern=pattern)
                if valid_move:
                    break
        
        # Else follow the pattern of the player before it and play a higher rank
        else:
            valid_move, pattern, choice, leading_rank = utils.smallest_valid_choice(hand=self.hand, pattern=pattern, leading_rank=leading_rank)
                
        # Return the card choice and subtract it from its hand
        if valid_move:
            choice = np.array(choice)
            self.hand -= choice
        return valid_move, pattern, choice, leading_rank, np.sum(self.hand)
    
    
    def claim_landlord(self, cards):
        """
        Randomly decide whether to claim the landlord cards.
        
        params:
            cards (array): The remaining cards in the deck [Unused]
            
        returns:
            landlord (bool): Whether the player claims the landlord cards
        """
        self.landlord = random.choice([True, False])
        return self.landlord
       
    
class UserPlayer(Player):
    def __init__(self):
        super().__init__()
        

    # The first card is the leading rank
    def select_action(self, state):
        """
        Asks the user to input a move based on the current state of the game.
        
        params:
            pattern      (str):  The pattern of the previous player
            prev_choice  (array): The cards played by the previous player
            leading_rank (int):  The rank of the leading card
            
        returns:
            valid_move      (bool): Whether the player made a valid move
            pattern         (str):  The pattern of the move
            choice          (array): The cards played by the player
            leading_rank    (int):  The rank of the leading card
            remainder       (int):  Number of cards remaining in the player's hand
        """
        if not state["action_history"]:
            pattern = None
            leading_rank = -1
        else:
            pattern = state["action_history"][-1]["pattern"]
            leading_rank = state["action_history"][-1]["leading_rank"]
        
        # Get the user input
        print(f"Hand: {utils.freq_array_to_card_str(self.hand)}")
        
        while True:
            valid_input = True
            if self.free:
                leading_rank = -1   # Reset the leading rank
                print("FREE TO MOVE")
                pattern = input("Enter the pattern: ")  # 1x5 format
                
                # Check if the pattern exists in the moveset
                if not pattern in self.moveset:
                    print("Invalid pattern. Please try again.")
                    continue
                
            # Assumes the first card is the leading rank
            user_cards = input("Enter your move: ")    # 334455 format
            
            # Check if all cards are known in the card set
            for c in user_cards:
                if c not in CARDS.values():
                    valid_input = False
            
            if valid_input:
                valid_move, pattern, choice, user_rank, valid_input = utils.read_user_cards(user_cards, pattern, leading_rank, self.hand)   # Convert to numpy frequency array
                
            # Escape the while loop only if the input is valid
            if not valid_input:
                print("Invalid card selection. Please try again.")
            else:
                break
        
        # After a successful move, the player is no longer free to move
        self.free = False
            
        # Record the play
        if valid_move:
            choice = np.array(choice)
            self.hand -= choice
            leading_rank = user_rank
        return valid_move, pattern, choice, leading_rank, np.sum(self.hand)
    

    def claim_landlord(self, cards):
        """
        Lets the user choose whether to claim the landlord cards.
        
        params:
            cards (array): The remaining cards in the deck
            
        returns:
            landlord (bool): Whether the player claims the landlord cards
        """
        print(f"\nCards in hand: {utils.freq_array_to_card_str(self.hand)}")
        print(f"Landlord cards: {utils.freq_array_to_card_str(cards)}")
        self.landlord = input("Claim the landlord cards? (y/n): ") == "y"
        return self.landlord
        

import torch
import torch.nn as nn
           
class RLPlayer(Player):
    # TODO
    def __init__(self, num_players, num_patterns, num_ranks, deck_moves, lstm_hidden_size=128, fc_hidden_size=256):
        super().__init__()  # Hand, moveset, landlord, free
        
        self.valid_move = False
        self.hand_mask = None
        self.curr_mask = None
        self.deck_moves = deck_moves
        self.num_actions = len(deck_moves)
        
        # Embedding layers for player ID, move patterns, and card ranks
        self.player_embedding = nn.Embedding(num_players, 8)
        self.pattern_embedding = nn.Embedding(num_patterns, 64)
        self.rank_embedding = nn.Embedding(num_ranks, 64)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=8 + 64 + 64, hidden_size=lstm_hidden_size, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, num_actions)
        
    # TODO
    def forward(self, player_id, move_pattern, card_rank, move_history):
        """
        Forward pass of the LSTM-DQN agent.
        
        Params:
            player_id: Tensor of player IDs for the current move (batch_size,)
            move_pattern: Tensor of move patterns (batch_size,)
            card_rank: Tensor of card ranks (batch_size,)
            move_history: Sliding window of past 10 moves (batch_size, sequence_length, input_size)
        
        Returns:
            Q-values for each action in the current state (batch_size, num_actions)
        """
        # Embedding lookups
        player_emb = self.player_embedding(player_id)  # (batch_size, 8)
        pattern_emb = self.pattern_embedding(move_pattern)  # (batch_size, 64)
        rank_emb = self.rank_embedding(card_rank)  # (batch_size, 64)

        # Concatenate the embeddings to form a single move representation
        move_emb = torch.cat([player_emb, pattern_emb, rank_emb], dim=-1)  # (batch_size, 136)

        # Pass the move history through the LSTM layer
        lstm_out, _ = self.lstm(move_history)  # (batch_size, sequence_length, lstm_hidden_size)
        lstm_last_hidden = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size)

        # Fully connected layers to predict Q-values
        x = torch.relu(self.fc1(lstm_last_hidden))  # (batch_size, fc_hidden_size)
        q_values = self.fc2(x)  # (batch_size, num_actions)

        return q_values
    
    # claimed (bool) = self.players[p].claim_landlord(card_freq)
    def claim_landlord(self, card_freq):
        # TODO: Use self.hand and card_freq to determine if agent should claim landlord
        pass
        
        
    def select_action(self, state, epsilon=0):
        prev_pattern = state["action_history"][-1]["pattern"]
        prev_leading_rank = state["action_history"][-1]["leading_rank"]
        
        # Agent is free to move if everyone else skipped
        if state["curr_skips"] >= self.num_players - 1:
            self.free = True
        
        self.valid_move, self.hand_mask, self.curr_mask = utils.get_hand_moves(self.hand, self.free, prev_pattern, prev_leading_rank, self.hand_mask, self.deck_moves)
        
        # No longer free to move after making a move
        self.free = False
        
        # No action selection if there are no valid moves
        if not self.valid_move:
            return None
        
        # TODO: Use the network to select an action from self.curr_mask
        # if random.random() < epsilon:
        #     return random.randint(0, agent.fc2.out_features - 1)
        # else:
        #     # Select action with the highest Q-value (exploitation)
        #     state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        #     with torch.no_grad():
        #         q_values = agent(state)
        #     return q_values.argmax().item()
        
        # TODO: return valid_move, pattern, prev_choice, leading_rank, remainder