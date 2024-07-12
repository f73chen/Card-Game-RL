import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOAgent(nn.Module):
    def __init__(self, input_dim, embedding_dim, action_dim):
        super(PPOAgent, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim + 1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.action_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, state):
        embedded = self.embedding(torch.arange(len(state)))
        combined = torch.cat((embedded, state.unsqueeze(1).float()), dim=1)
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_probs, state_values

    def select_action(self, state):
        action_probs, _ = self.forward(state)
        m = Categorical(action_probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

# Training loop (simplified)
def train(env, agent, optimizer, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        values = []

        while True:
            action, log_prob = agent.select_action(state)
            next_state, reward, done = env.step(action)
            _, value = agent(state)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)

            state = next_state
            if done:
                break

        # Compute returns and losses
        returns = compute_returns(rewards)
        loss = compute_loss(log_probs, returns, values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns)

def compute_loss(log_probs, returns, values):
    advantage = returns - torch.cat(values).squeeze()
    action_loss = -torch.cat(log_probs) * advantage.detach()
    value_loss = advantage.pow(2)
    return action_loss.sum() + value_loss.sum()

# Setup
card_ranks = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, '10': 7, 'J': 8, 'Q': 9, 'K': 10, 'A': 11, '2': 12}
embedding_dim = 5
action_dim = 4  # Example action space size
env = CardGameEnv()
agent = PPOAgent(len(card_ranks), embedding_dim, action_dim)
optimizer = optim.Adam(agent.parameters(), lr=0.01)

# Train the agent
train(env, agent, optimizer, num_episodes=1000)
