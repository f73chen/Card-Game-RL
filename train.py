import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from main import GameEnv

# TODO: Rewrite this whole page --> LSTM-DQN

# Define an enhanced RNN-DQN model with embedding layers and attention mechanism
class RNN_DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128, num_layers=1, embedding_dim=64):
        super(RNN_DQN, self).__init__()
        # Embeddings for moveset and card ranks
        self.moveset_embedding = nn.Embedding(num_embeddings=50, embedding_dim=embedding_dim)  # Assuming max 50 move patterns
        self.rank_embedding = nn.Embedding(num_embeddings=15, embedding_dim=embedding_dim)  # 15 possible ranks

        # LSTM layers
        self.lstm = nn.LSTM(state_size + 2 * embedding_dim, hidden_size, num_layers, batch_first=True)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        
        # Fully connected layer for final action selection
        self.fc = nn.Linear(hidden_size, action_size)

    def forward(self, x, moves, ranks):
        # Embedding the moveset and ranks
        moves_embedded = self.moveset_embedding(moves)
        ranks_embedded = self.rank_embedding(ranks)

        # Concatenating state with embeddings
        x = torch.cat((x, moves_embedded, ranks_embedded), dim=-1)

        # Passing through LSTM
        h0 = torch.zeros(1, x.size(0), 128).to(x.device)
        c0 = torch.zeros(1, x.size(0), 128).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        # Applying attention mechanism
        attn_output, _ = self.attention(out, out, out)

        # Passing through fully connected layer
        out = self.fc(attn_output[:, -1, :])
        return out

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def train_rnn_dqn(env, num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay):
    state_size = env.state_size
    action_size = env.action_size
    rnn_dqn = RNN_DQN(state_size, action_size)
    target_rnn_dqn = RNN_DQN(state_size, action_size)
    target_rnn_dqn.load_state_dict(rnn_dqn.state_dict())
    optimizer = optim.Adam(rnn_dqn.parameters())
    replay_buffer = ReplayBuffer(buffer_size=10000)
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            if random.random() < epsilon:
                action = random.choice(range(action_size))
            else:
                with torch.no_grad():
                    action = torch.argmax(rnn_dqn(torch.tensor(state, dtype=torch.float32).unsqueeze(0))).item()

            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = rnn_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = target_rnn_dqn(next_states).max(1)[0]
                target_q_values = rewards + (gamma * next_q_values * (1 - dones))

                loss = nn.MSELoss()(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 10 == 0:
            target_rnn_dqn.load_state_dict(rnn_dqn.state_dict())

        epsilon = max(epsilon_end, epsilon_decay * epsilon)

    return rnn_dqn

# Example usage:
env = GameEnv(num_decks=1, num_players=3, mode="lord", players=[])
trained_rnn_dqn = train_rnn_dqn(env, num_episodes=1000, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995)

# After training
torch.save(trained_rnn_dqn.state_dict(), 'rnn_dqn_model.pth')
