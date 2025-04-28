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
        action_dim: int,
        action_num: int,
    ) -> None:
        super().__init__()
        self.env_dim = env_dim
        self.action_num = action_num
        self.input_dim = env_dim + action_dim
        self.output_dim = 1

        layers = [
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim),
        ]
        self.model = nn.Sequential(*layers)
        # self.train()

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

    def total_reward(self) -> float:
        return sum(experience.reward for experience in self.buffer)


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

    def get_buffer_total_reward(self) -> float:
        return self._replay_buffer.total_reward()

    def act(self, observation: np.floating, epsilon: float) -> tuple[int, float | None]:
        if np.random.rand() < epsilon:
            action = np.random.randint(self._action_num)
            return action, None
        x = torch.from_numpy(observation.astype(np.float32))
        max_q = torch.tensor(-np.inf)
        with torch.no_grad():
            best_action = 0
            for action in range(self._action_num):
                x_ = torch.cat((x, torch.tensor([action], dtype=torch.float32)))
                q = self._policy_net(x_)
                if q > max_q:
                    max_q = q
                    best_action = action
        return best_action, max_q.item()

    def policy_update(self, gamma: float) -> float:
        experiences = self._replay_buffer.sample()

        observations = torch.tensor([experience.observation for experience in experiences])
        actions = torch.tensor([experience.action for experience in experiences], dtype=torch.float32)
        next_observations = torch.tensor([experience.next_observation for experience in experiences])
        next_actions = torch.tensor([experience.next_action for experience in experiences])
        rewards = torch.tensor([experience.reward for experience in experiences]).unsqueeze(1)
        terminated = torch.tensor(
            [experience.terminated for experience in experiences],
            dtype=torch.float32,
        ).unsqueeze(1)

        # q preds
        x_tensor = torch.cat((observations, actions.unsqueeze(1)), dim=1)
        q_preds = self._policy_net(x_tensor)

        with torch.no_grad():
            x_tensor = torch.cat((next_observations, next_actions.unsqueeze(1)), dim=1)
            next_q_preds = self._policy_net(x_tensor)
            q_targets = rewards + gamma * (1 - terminated) * next_q_preds
        loss = torch.nn.functional.mse_loss(q_preds, q_targets)
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
    learning_rate: float = 0.002
    log_wandb: bool = False


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


class SARSATrainer:
    def __init__(self, config: Config) -> None:
        self.env = gym.make(config.env.env_name, render_mode=config.env.render_mode)
        if isinstance(self.env.action_space, gym.spaces.Discrete) is False:
            raise ValueError("Only discrete action space is supported.")
        env_dim = self.env.observation_space.shape[0]  # type: ignore[index]
        action_num = self.env.action_space.n  # type: ignore[attr-defined]
        policy_net = PolicyNet(env_dim=env_dim, action_dim=1, action_num=action_num)
        optimizer = optim.RMSprop(policy_net.parameters(), lr=config.train.learning_rate)
        self.agent = Agent(policy_net=policy_net, optimizer=optimizer)

        self.num_episodes = config.train.num_episodes
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

    def train(self) -> None:
        epsilon = 1.0
        for episode in range(self.num_episodes):
            observation, _ = self.env.reset()
            terminated, truncated = False, False
            q_values = []
            while not (terminated or truncated):
                action, q_value = self.agent.act(observation, epsilon)
                if q_value is not None:
                    q_values.append(q_value)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
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
            loss = self.agent.policy_update(gamma=self.gamma)
            total_reward = self.agent.get_buffer_total_reward()
            solved = total_reward > self.solved_threshold
            self.agent.onpolicy_reset()
            epsilon = max(0.01, epsilon * 0.997)
            q_value_mean = np.mean(q_values)

            print(
                f"Episode {episode}, epsilon: {epsilon}, loss: {loss}, q_value_mean: {q_value_mean}, "
                f"total_reward: {total_reward}, solved: {solved}"
            )
            if self.config.train.log_wandb:
                wandb.log(
                    {
                        "episode": episode,
                        "loss": loss,
                        "q_value_mean": q_value_mean,
                        "total_reward": total_reward,
                    }
                )


if __name__ == "__main__":
    default_config = Config(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None, solved_threshold=475.0),
        train=TrainConfig(num_episodes=100000, learning_rate=0.01, log_wandb=True),
    )
    trainer = SARSATrainer(default_config)
    trainer.train()
