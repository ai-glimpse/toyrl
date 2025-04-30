from dataclasses import asdict, dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb


class PolicyNet(nn.Module):
    def __init__(self, env_dim: int, action_num: int) -> None:
        super().__init__()
        layers = [
            nn.Linear(env_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_num),
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
    def __init__(self, policy_net: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self._policy_net = policy_net
        self._optimizer = optimizer
        self._replay_buffer = ReplayBuffer()

    def onpolicy_reset(self) -> None:
        self._replay_buffer.reset()

    def add_experience(self, experience: Experience) -> None:
        self._replay_buffer.add_experience(experience)

    def get_buffer_total_reward(self) -> float:
        return self._replay_buffer.total_reward()

    def act(self, observation: np.floating) -> tuple[int, torch.Tensor]:
        x = torch.from_numpy(observation.astype(np.float32))
        logits = self._policy_net(x)
        next_action_dist = torch.distributions.Categorical(logits=logits)
        action = next_action_dist.sample()
        action_log_prob = next_action_dist.log_prob(action)
        return action.item(), action_log_prob

    def policy_update(self, gamma: float, with_baseline: bool) -> float:
        experiences = self._replay_buffer.sample()
        # returns
        T = len(experiences)
        returns = torch.zeros(T)
        future_ret = 0.0
        for t in reversed(range(T)):
            future_ret = experiences[t].reward + gamma * future_ret
            returns[t] = future_ret
        # baseline
        if with_baseline:
            returns -= returns.mean()

        # log_probs
        action_log_probs = [exp.action_log_prob for exp in experiences]
        log_probs = torch.stack(action_log_probs)
        # loss
        loss = -log_probs * returns
        loss = torch.sum(loss)
        # update
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.item()


@dataclass
class EnvConfig:
    env_name: str = "CartPole-v1"
    render_mode: str | None = None
    solved_threshold: float = 475.0


@dataclass
class TrainConfig:
    gamma: float = 0.999
    num_episodes: int = 500
    learning_rate: float = 0.01
    with_baseline: bool = True
    log_wandb: bool = False


@dataclass
class ReinforceConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


class ReinforceTrainer:
    def __init__(self, config: ReinforceConfig) -> None:
        self.config = config
        self.env = gym.make(config.env.env_name, render_mode=config.env.render_mode)
        env_dim = self.env.observation_space.shape[0]  # type: ignore[index]
        action_num = self.env.action_space.n  # type: ignore[attr-defined]
        policy_net = PolicyNet(env_dim=env_dim, action_num=action_num)
        optimizer = optim.Adam(policy_net.parameters(), lr=config.train.learning_rate)
        self.agent = Agent(policy_net=policy_net, optimizer=optimizer)

        self.num_episodes = config.train.num_episodes
        self.gamma = config.train.gamma
        self.with_baseline = config.train.with_baseline
        self.solved_threshold = config.env.solved_threshold
        if config.train.log_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="Reinforce",
                name=f"[{config.env.env_name}]lr={config.train.learning_rate}, baseline={config.train.with_baseline}",
                # track hyperparameters and run metadata
                config=asdict(config),
            )

    def train(self) -> None:
        for epi in range(self.num_episodes):
            observation, _ = self.env.reset()
            terminated, truncated = False, False
            while not (terminated or truncated):
                action, action_log_prob = self.agent.act(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                experience = Experience(
                    observation=observation,
                    action=action,
                    action_log_prob=action_log_prob,
                    reward=float(reward),
                    terminated=terminated,
                    truncated=truncated,
                    next_observation=next_observation,
                )
                self.agent.add_experience(experience)
                observation = next_observation
                if self.config.env.render_mode is not None:
                    self.env.render()
                self.env.render()
            loss = self.agent.policy_update(
                gamma=self.gamma,
                with_baseline=self.with_baseline,
            )
            total_reward = self.agent.get_buffer_total_reward()
            solved = total_reward > self.solved_threshold
            self.agent.onpolicy_reset()
            print(f"Episode {epi}, loss: {loss}, total_reward: {total_reward}, solved: {solved}")
            if self.config.train.log_wandb:
                wandb.log(
                    {
                        "episode": epi,
                        "loss": loss,
                        "total_reward": total_reward,
                    }
                )


if __name__ == "__main__":
    default_config = ReinforceConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None, solved_threshold=475.0),
        train=TrainConfig(num_episodes=100000, learning_rate=0.002, with_baseline=True, log_wandb=True),
    )
    trainer = ReinforceTrainer(default_config)
    trainer.train()
