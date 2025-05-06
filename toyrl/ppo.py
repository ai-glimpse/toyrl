from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


class ActorPolicyNet(nn.Module):
    def __init__(self, env_dim: int, action_num: int) -> None:
        super().__init__()
        layers = [
            nn.Linear(env_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_num),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # type: ignore


class CriticValueNet(nn.Module):
    def __init__(self, env_dim: int) -> None:
        super().__init__()
        layers = [
            nn.Linear(env_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # type: ignore


@dataclass
class Experience:
    terminated: bool
    truncated: bool
    observation: Any
    action: Any
    reward: float
    next_observation: Any


@dataclass
class ReplayBuffer:
    buffer: list[Experience] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.buffer)

    def add_experience(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def reset(self) -> None:
        self.buffer = []

    # TODO: try other sampling methods?
    def sample(self) -> list[Experience]:
        return self.buffer


@dataclass
class EnvConfig:
    env_name: str = "CartPole-v1"
    render_mode: str | None = None
    solved_threshold: float = 475.0


@dataclass
class TrainConfig:
    gamma: float = 0.999
    lambda_: float = 0.98

    total_timesteps: int = 500000
    num_envs: int = 4
    """The number of parallel game environments"""
    batch_size: int = 64
    update_epochs: int = 4  # K
    """The K epochs to update the policy"""
    num_minibatches: int = 4
    """The number of mini-batches"""

    learning_rate: float = 2.5e-4
    log_wandb: bool = False


@dataclass
class PPOConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


class PPOAgent:
    def __init__(self, actor: ActorPolicyNet, critic: CriticValueNet, optimizer: optim.Optimizer) -> None:
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer


class PPOTrainer:
    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.envs = self._make_env()
        env_dim = self.envs.observation_space.shape[0]  # type: ignore[index]
        action_num = self.envs.action_space.n  # type: ignore[attr-defined]

        actor = ActorPolicyNet(env_dim=env_dim, action_num=action_num)
        critic = CriticValueNet(env_dim=env_dim)
        optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=config.train.learning_rate
        )
        self.agent = PPOAgent(actor=actor, critic=critic, optimizer=optimizer)

    def _make_env(self):
        envs = gym.make_vec(
            id=self.config.env.env_name,
            num_envs=self.config.train.num_envs,
            render_mode=self.config.env.render_mode,
            autoreset_mode=gym.vector.vector_env.AutoresetMode.NEXT_STEP,
            wrappers=[
                gym.wrappers.RecordEpisodeStatistics,
            ],
        )
        return envs

    def train(self):
        pass
