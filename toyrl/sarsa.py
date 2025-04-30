from dataclasses import asdict, dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb


class PolicyNet(nn.Module):
    def __init__(
        self,
        env_dim: int,
        action_num: int,
    ) -> None:
        super().__init__()
        self.env_dim = env_dim
        self.action_num = action_num

        layers = [
            nn.Linear(self.env_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_num),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # type: ignore


@dataclass
class Experience:
    terminated: bool
    truncated: bool
    observation: Any  # S
    action: Any  # A
    reward: float  # R
    next_observation: Any = None  # S'
    next_action: Any = None  # A'


@dataclass
class ReplayBuffer:
    buffer: list[Experience] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.buffer)

    def add_experience(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def reset(self) -> None:
        self.buffer = []

    def sample(self, with_next_sa: bool = True) -> list[Experience]:
        if with_next_sa is False:
            return self.buffer
        else:
            res = []
            for i in range(len(self.buffer) - 1):
                experience = self.buffer[i]
                next_experience = self.buffer[i + 1]
                res.append(
                    Experience(
                        observation=experience.observation,
                        action=experience.action,
                        reward=experience.reward,
                        next_observation=next_experience.observation,
                        next_action=next_experience.action,
                        terminated=next_experience.terminated,
                        truncated=next_experience.truncated,
                    )
                )
            return res


class Agent:
    def __init__(self, policy_net: PolicyNet, optimizer: torch.optim.Optimizer) -> None:
        self._policy_net = policy_net
        self._optimizer = optimizer
        self._replay_buffer = ReplayBuffer()
        self._action_num = policy_net.action_num

    def onpolicy_reset(self) -> None:
        self._replay_buffer.reset()

    def add_experience(self, experience: Experience) -> None:
        self._replay_buffer.add_experience(experience)

    def act(self, observation: np.floating, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            action = np.random.randint(self._action_num)
            return action
        x = torch.from_numpy(observation.astype(np.float32))
        logits = self._policy_net(x)
        action = int(torch.argmax(logits).item())
        return action

    def policy_update(self, gamma: float) -> float:
        experiences = self._replay_buffer.sample()

        observations = torch.tensor([experience.observation for experience in experiences])
        actions = torch.tensor([experience.action for experience in experiences], dtype=torch.float32)
        next_observations = torch.tensor([experience.next_observation for experience in experiences])
        next_actions = torch.tensor([experience.next_action for experience in experiences])
        rewards = torch.tensor([experience.reward for experience in experiences])
        terminated = torch.tensor(
            [experience.terminated for experience in experiences],
            dtype=torch.float32,
        )

        # q preds
        action_q_preds = self._policy_net(observations).gather(1, actions.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_action_q_preds = self._policy_net(next_observations).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_targets = rewards + gamma * (1 - terminated) * next_action_q_preds
        loss = torch.nn.functional.mse_loss(action_q_preds, q_targets)
        # update
        self._optimizer.zero_grad()
        loss.backward()
        # clip grad
        torch.nn.utils.clip_grad_norm_(self._policy_net.parameters(), max_norm=1.0)
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
    max_training_steps: int = 500000
    """The maximum number of environment steps to train for."""
    learning_rate: float = 2.5e-4
    """The learning rate for the optimizer."""
    log_wandb: bool = False
    """Whether to log the training process to Weights and Biases."""


@dataclass
class SarsaConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


class SarsaTrainer:
    def __init__(self, config: SarsaConfig) -> None:
        self.config = config
        self.env = self._make_env()
        if isinstance(self.env.action_space, gym.spaces.Discrete) is False:
            raise ValueError("Only discrete action space is supported.")
        env_dim = self.env.observation_space.shape[0]  # type: ignore[index]
        action_num = self.env.action_space.n  # type: ignore[attr-defined]
        policy_net = PolicyNet(env_dim=env_dim, action_num=action_num)
        optimizer = optim.Adam(policy_net.parameters(), lr=config.train.learning_rate)
        self.agent = Agent(policy_net=policy_net, optimizer=optimizer)

        self.gamma = config.train.gamma
        self.solved_threshold = config.env.solved_threshold
        if config.train.log_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="SARSA",
                name=f"[{config.env.env_name}],lr={config.train.learning_rate}",
                # track hyperparameters and run metadata
                config=asdict(config),
            )

    def _make_env(self):
        env = gym.make(self.config.env.env_name, render_mode=self.config.env.render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.Autoreset(env)
        return env

    def train(self) -> None:
        epsilon = 1.0
        global_step = 0

        observation, _ = self.env.reset()
        while global_step < self.config.train.max_training_steps:
            global_step += 1
            epsilon = max(0.05, epsilon * 0.9999)

            action = self.agent.act(observation, epsilon)
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            experience = Experience(
                observation=observation,
                action=action,
                reward=float(reward),
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
            )
            self.agent.add_experience(experience)
            observation = next_observation
            if self.env.render_mode is not None:
                self.env.render()

            if terminated or truncated:
                if info and "episode" in info:
                    episode_reward = info["episode"]["r"]
                    loss = self.agent.policy_update(gamma=self.gamma)
                    self.agent.onpolicy_reset()
                    print(
                        f"global_step={global_step}, epsilon={epsilon}, episodic_return={episode_reward}, loss={loss}"
                    )
                    if self.config.train.log_wandb:
                        wandb.log(
                            {
                                "global_step": global_step,
                                "episode_reward": episode_reward,
                                "loss": loss,
                            }
                        )


if __name__ == "__main__":
    default_config = SarsaConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None, solved_threshold=475.0),
        train=TrainConfig(
            max_training_steps=2000000,
            learning_rate=0.01,
            log_wandb=True,
        ),
    )
    trainer = SarsaTrainer(default_config)
    trainer.train()
