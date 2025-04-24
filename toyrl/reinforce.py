import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # type: ignore


class Agent:
    def __init__(self, policy_net: nn.Module) -> None:
        self.policy_net = policy_net
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []

    def onpolicy_reset(self) -> None:
        self.log_probs = []
        self.rewards = []

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        logits = self.policy_net(x)
        next_action_dist = torch.distributions.Categorical(logits=logits)
        action = next_action_dist.sample()
        log_prob = next_action_dist.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()


def train(agent: Agent, optimizer: torch.optim.Optimizer, gamma: float = 0.99):
    T = len(agent.rewards)
    rets = torch.zeros(T)
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = agent.rewards[t].cpu().item() + gamma * future_ret
        rets[t] = future_ret
    log_probs = torch.stack(agent.log_probs)
    loss = -log_probs * rets
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def main():
    # env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("CartPole-v1")
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n
    policy_net = PolicyNet(in_dim, out_dim)
    agent = Agent(policy_net)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
    for epi in range(500):
        state, _ = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(torch.tensor(reward))
            env.render()
        loss = train(agent, optimizer)
        total_reward = sum(agent.rewards)
        solved = total_reward > 475.0
        agent.onpolicy_reset()
        print(f"Episode {epi}, loss: {loss}, total_reward: {total_reward}, solved: {solved}")


if __name__ == "__main__":
    main()
