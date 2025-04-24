from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class Experience:
    observation: Any
    action: Any
    action_log_prob: torch.Tensor
    reward: float
    next_observation: Any
    terminated: bool
    truncated: bool


@dataclass
class ReplayBuffer:
    buffer: list[Experience] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.buffer)

    def add_experience(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def reset(self) -> None:
        self.buffer = []

    def sample(self) -> list[Experience]:
        return self.buffer

    def total_reward(self) -> float:
        return sum(experience.reward for experience in self.buffer)


class Agent:
    def __init__(self, policy_net: nn.Module) -> None:
        self.policy_net = policy_net
        self.replay_buffer = ReplayBuffer()

    def onpolicy_reset(self) -> None:
        self.replay_buffer.reset()

    def act(self, observation) -> tuple[int, torch.Tensor]:
        x = torch.from_numpy(observation.astype(np.float32))
        logits = self.policy_net(x)
        next_action_dist = torch.distributions.Categorical(logits=logits)
        action = next_action_dist.sample()
        action_log_prob = next_action_dist.log_prob(action)
        return action.item(), action_log_prob


def train(agent: Agent, optimizer: torch.optim.Optimizer, gamma: float):
    experiences = agent.replay_buffer.sample()
    # returns
    T = len(experiences)
    returns = torch.zeros(T)
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = experiences[t].reward + gamma * future_ret
        returns[t] = future_ret
    # log_probs
    action_log_probs = [exp.action_log_prob for exp in experiences]
    log_probs = torch.stack(action_log_probs)
    # loss
    loss = -log_probs * returns
    loss = torch.sum(loss)
    # update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def main(gamma: float = 0.99) -> None:
    # env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("CartPole-v1")
    in_dim = env.observation_space.shape[0]  # type: ignore[index]
    out_dim = env.action_space.n  # type: ignore[attr-defined]
    policy_net = PolicyNet(in_dim, out_dim)
    agent = Agent(policy_net)
    optimizer = optim.Adam(agent.policy_net.parameters(), lr=0.01)
    for epi in range(500):
        observation, _ = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action, action_log_prob = agent.act(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            experience = Experience(
                observation=observation,
                action=action,
                action_log_prob=action_log_prob,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                next_observation=next_observation,
            )
            agent.replay_buffer.add_experience(experience)
            observation = next_observation
            env.render()
        loss = train(agent, optimizer, gamma)
        total_reward = agent.replay_buffer.total_reward()
        solved = total_reward > 475.0
        agent.onpolicy_reset()
        print(f"Episode {epi}, loss: {loss}, total_reward: {total_reward}, solved: {solved}")


if __name__ == "__main__":
    main()
