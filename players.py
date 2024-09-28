import numpy as np
import random
import utils
import torch
import torch.nn as nn

from consts import *

class Player:
    def __init__(self, num_players):
        # Don't reset these
        self.num_players = num_players
        self.deck_moves = None
        
        # Do reset these
        self.hand = None
        self.landlord = False
        self.free = False
        
        self.hand_mask = None
        self.curr_mask = None
        
    # Reset everything except the moveset and deck_moves
    def reset(self):
        self.hand = None
        self.landlord = False
        self.free = False
        
        self.hand_mask = None
        self.curr_mask = None
        
    # Default: Select the lowest available action unless player must skip
    def select_action(self, state, landlord_cards=None):
        # Player becomes free to move if everyone else skipped
        if state["curr_skips"] >= self.num_players - 1:
            self.free = True
            
        # If free to move, can choose any pattern
        if self.free:
            prev_pattern = None
            prev_leading_rank = -1
        
        # If not free to move, must follow previous pattern
        else:
            prev_pattern = state["action_history"][-1]["pattern"]
            prev_leading_rank = state["action_history"][-1]["leading_rank"]
        
        # Get the mask of available actions
        self.hand_mask, self.curr_mask = utils.get_hand_moves(self.hand, self.free, prev_pattern, prev_leading_rank, self.hand_mask, self.deck_moves, state["choosing_landlord"])
        self.free = False
            
        # Action selection logic
        # If skipping is the only available action, must skip
        if sum(self.curr_mask) == 1:
            pattern, choice, leading_rank = self.deck_moves[-1]
            
        # Else randomly select a pattern and choose the first (smallest) option
        else:
            available_actions = self.deck_moves[self.curr_mask]
            available_patterns = set([action[0] for action in available_actions])
            pattern = np.random.choice(available_patterns)
            for action in available_actions:
                if action[0] == pattern:
                    pattern, choice, leading_rank = action
                    
        self.hand -= np.array(choice)
            
        return pattern, choice, leading_rank, np.sum(self.hand)
    
    
class UserPlayer(Player):
    # Note: Inherits __init__ and reset()
    def select_action(self, state, landlord_cards=None):
        # Player becomes free to move if everyone else skipped
        if state["curr_skips"] >= self.num_players - 1:
            self.free = True
            
        # If free to move, can choose any pattern
        if self.free:
            prev_pattern = None
            prev_leading_rank = -1
        
        # If not free to move, must follow previous pattern
        else:
            prev_pattern = state["action_history"][-1]["pattern"]
            prev_leading_rank = state["action_history"][-1]["leading_rank"]
        
        # Get the mask of available actions
        self.hand_mask, self.curr_mask = utils.get_hand_moves(self.hand, self.free, prev_pattern, prev_leading_rank, self.hand_mask, self.deck_moves, state["choosing_landlord"])
            
        # Action selection logic
        # If skipping is the only available action, must skip
        if sum(self.curr_mask) == 1:
            pattern, choice, leading_rank = self.deck_moves[-1]
            print("No playable moves, automatically skipped")
            
        # Else let the user choose an action and validate it
        else:
            available_actions = self.deck_moves[self.curr_mask]
            available_patterns = set([action[0] for action in available_actions])
            
            if state["choosing_landlord"]:                
                print(f"\nCards in hand: {utils.freq_array_to_card_str(self.hand)}")
                print(f"Landlord cards: {utils.freq_array_to_card_str(landlord_cards)}")
                claimed = input("Claim the landlord cards? (y/n): ") == "y"
                
                if claimed:
                    pattern, choice, leading_rank = [action for action in available_actions if action[0] == "claim_landlord"][0]
                else:
                    pattern, choice, leading_rank = [action for action in available_actions if action[0] == "refuse_landlord"][0]
            
            else:
                print(f"Hand: {utils.freq_array_to_card_str(self.hand)}")
                while True:
                    valid_input = True
                    if self.free:
                        print("FREE TO MOVE")
                        
                    # Note: Always ask for the pattern in case of bomb
                    pattern = input("Enter the pattern: ")
                    if not pattern:
                        pattern = prev_pattern
                    
                    # Validate the pattern
                    if pattern not in available_patterns:
                        print("Invalid pattern. Please try again.")
                        print(f"Available patterns: {available_patterns}")
                        continue
                        
                    # Assume the first card is the leading rank
                    user_cards = input("Enter your move: ")
                    
                    # Check if all inputs represent an existing card
                    unknown_cards = []
                    for c in user_cards:
                        if c not in CARDS.values():
                            unknown_cards.append(c)
                    if unknown_cards:
                        print(f"{unknown_cards} is not a valid card. Please try again.")
                        continue
                    
                    # Can't skip when free
                    if self.free and not user_cards:
                        print("Cannot skip when free to move. Please try again.")
                        continue
                    
                    # Assume that the pattern and individual cards are already valid
                    pattern, choice, leading_rank, valid_choice = utils.read_user_cards(pattern, user_cards, available_actions)
                            
                    # Escape the while loop only if the input is valid
                    if not valid_choice:
                        print("Invalid card selection. Please try again.")
                    else:
                        break
                    
                # After a successful move, the player is no longer free to move
                self.free = False
                    
        self.hand -= np.array(choice).astype(int)
        return pattern, choice, leading_rank, np.sum(self.hand)
        
    
    
class RLPlayer(Player):
    # TODO
    def __init__(self, num_players, num_patterns, num_ranks, lstm_hidden_size=128, fc_hidden_size=256):
        super().__init__(num_players)
        
        self.num_total_actions = len(self.deck_moves)
        
        # Embedding layers for player ID, move patterns, and card ranks
        self.player_embedding = nn.Embedding(num_players, 8)
        self.pattern_embedding = nn.Embedding(num_patterns, 64)
        self.rank_embedding = nn.Embedding(num_ranks, 64)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=8 + 64 + 64, hidden_size=lstm_hidden_size, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, self.num_total_actions)
    
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
    
    # TODO
    def select_action(self, state, epsilon=0):
        
        # TODO: Use the network to select an action from self.curr_mask
        # if random.random() < epsilon:
        #     return random.randint(0, agent.fc2.out_features - 1)
        # else:
        #     # Select action with the highest Q-value (exploitation)
        #     state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        #     with torch.no_grad():
        #         q_values = agent(state)
        #     return q_values.argmax().item()
        
        # TODO: If move is valid, update remainder
        
        # TODO: return valid_move, pattern, prev_choice, leading_rank, remainder
        pass