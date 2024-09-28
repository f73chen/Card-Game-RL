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
        
    # TODO: Select the lowest available action unless must skip
    def select_action(self, state):
        # When claiming landlord cards, only have 2 actions (claim/refuse)
        if state["choosing_landlord"]:
            self.curr_mask = [False] * len(self.deck_moves)
            self.curr_mask[-3] = True   # claim_landlord
            self.curr_mask[-2] = True   # refuse_landlord
            
        # Normal play
        else:
            # Agent becomes free to move if everyone else skipped
            if state["curr_skips"] >= self.num_players - 1:
                self.free = True
                
            # Free to move, can choose any pattern
            if self.free:
                prev_pattern = None
                prev_leading_rank = -1
            
            # Not free to move, must follow previous pattern
            else:
                prev_pattern = state["action_history"][-1]["pattern"]
                prev_leading_rank = state["action_history"][-1]["leading_rank"]
                
            self.hand_mask, self.curr_mask = utils.get_hand_moves(self.hand, self.free, prev_pattern, prev_leading_rank, self.hand_mask, self.deck_moves)
            self.free = False
            
        # Action selection logic
        
        
        
        # if pattern != "skip":
        #     # TODO: Make sure choice is a numpy array
        #     self.hand -= choice
            
        # return pattern, choice, leading_rank, np.sum(self.hand)
    
    
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