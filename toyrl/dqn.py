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
    replay_buffer_size: int = 100000
    buffer: list[Experience] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.buffer)

    def add_experience(self, experience: Experience) -> None:
        if len(self.buffer) >= self.replay_buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def reset(self) -> None:
        self.buffer = []

    def sample(self, batch_size: int) -> list[Experience]:
        real_batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), real_batch_size, replace=False)
        return [self.buffer[i] for i in indices]


class Agent:
    def __init__(
        self,
        policy_net: PolicyNet,
        target_net: PolicyNet | None,
        optimizer: torch.optim.Optimizer,
        replay_buffer_size: int,
    ) -> None:
        self._policy_net = policy_net
        self._target_net = target_net
        self._optimizer = optimizer
        self._replay_buffer = ReplayBuffer(replay_buffer_size)
        self._action_num = policy_net.action_num

    def add_experience(self, experience: Experience) -> None:
        self._replay_buffer.add_experience(experience)

    def act(self, observation: np.floating, tao: float) -> tuple[int, float]:
        x = torch.from_numpy(observation.astype(np.float32))
        logits = torch.empty(self._action_num)
        with torch.no_grad():
            for action in range(self._action_num):
                x_ = torch.cat((x, torch.tensor([action], dtype=torch.float32)))
                logits[action] = self._policy_net(x_) / tao
        next_action_prob = logits.softmax(dim=0)
        next_action = torch.distributions.Categorical(probs=next_action_prob).sample().item()
        q_value = logits[next_action].item()
        return next_action, q_value

    def sample(self, batch_size: int) -> list[Experience]:
        return self._replay_buffer.sample(batch_size)

    def policy_update(self, gamma: float, experiences: list[Experience]) -> float:
        observations = torch.tensor([experience.observation for experience in experiences])
        actions = torch.tensor([experience.action for experience in experiences], dtype=torch.float32)
        next_observations = torch.tensor([experience.next_observation for experience in experiences])
        rewards = torch.tensor([experience.reward for experience in experiences]).unsqueeze(1)
        terminated = torch.tensor(
            [experience.terminated for experience in experiences],
            dtype=torch.float32,
        ).unsqueeze(1)

        # q preds
        x_tensor = torch.cat((observations, actions.unsqueeze(1)), dim=1)
        q_preds = self._policy_net(x_tensor)

        action_probs = torch.empty(size=(len(experiences), self._action_num))
        with torch.no_grad():
            for next_action in range(self._action_num):
                next_actions = torch.tensor([next_action] * len(experiences), dtype=torch.float32)
                x_tensor = torch.cat((next_observations, next_actions.unsqueeze(1)), dim=1)
                action_probs[:, next_action] = self._policy_net(x_tensor).squeeze(1)

        next_actions = torch.argmax(action_probs, dim=1)
        x_tensor = torch.cat((next_observations, next_actions.unsqueeze(1)), dim=1)
        if self._target_net is None:  # Vanilla DQN
            next_q_preds = self._policy_net(x_tensor)
        else:  # Double DQN
            next_q_preds = self._target_net(x_tensor)
        q_targets = rewards + gamma * (1 - terminated) * next_q_preds
        loss = torch.nn.functional.mse_loss(q_preds, q_targets)
        # update
        self._optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self._policy_net.parameters(), 10)
        self._optimizer.step()
        return loss.item()

    def polyak_update(self, beta: float) -> None:
        if self._target_net is not None:
            for target_param, param in zip(self._target_net.parameters(), self._policy_net.parameters()):
                target_param.data.copy_(beta * target_param.data + (1 - beta) * param.data)
        else:
            raise ValueError("Target net is not set.")


@dataclass
class EnvConfig:
    env_name: str = "CartPole-v1"
    render_mode: str | None = None
    solved_threshold: float = 475.0


@dataclass
class TrainConfig:
    gamma: float = 0.999
    """The discount factor for future rewards."""
    replay_buffer_size: int = 10000  # K
    """The size of the replay buffer."""

    num_episodes: int = 500
    """The number of episodes to train the agent."""
    batch_pre_train: int = 4  # B
    """The number of batches to pre-train the agent."""
    batch_size: int = 32  # N
    """The size of each batch."""
    update_per_batch: int = 4  # U
    """The number of updates per batch."""

    learning_rate: float = 0.01
    """The learning rate for the optimizer."""

    with_target_net: bool = False
    """Whether to use a target network for training."""
    target_net_update_freq: int = 10  # F
    """The frequency of updating the target network."""
    beta: float = 0.0
    """The target network update rate(Polyak update)."""

    log_wandb: bool = False
    """Whether to log the training process to Weights and Biases."""


@dataclass
class DqnConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


class DqnTrainer:
    def __init__(self, config: DqnConfig) -> None:
        self.config = config
        self.env = gym.make(config.env.env_name, render_mode=config.env.render_mode)
        if isinstance(self.env.action_space, gym.spaces.Discrete) is False:
            raise ValueError("Only discrete action space is supported.")
        env_dim = self.env.observation_space.shape[0]  # type: ignore[index]
        action_num = self.env.action_space.n  # type: ignore[attr-defined]

        policy_net = PolicyNet(env_dim=env_dim, action_dim=1, action_num=action_num)
        optimizer = optim.Adam(policy_net.parameters(), lr=config.train.learning_rate)
        if config.train.with_target_net:
            target_net = PolicyNet(env_dim=env_dim, action_dim=1, action_num=action_num)
            target_net.load_state_dict(policy_net.state_dict())
        else:
            target_net = None
        self.agent = Agent(
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            replay_buffer_size=config.train.replay_buffer_size,
        )

        self.num_episodes = config.train.num_episodes
        self.gamma = config.train.gamma
        self.solved_threshold = config.env.solved_threshold
        if config.train.log_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project=self._get_dqn_name(),
                name=f"[{config.env.env_name}],lr={config.train.learning_rate}",
                # track hyperparameters and run metadata
                config=asdict(config),
            )

    def _get_dqn_name(self) -> str:
        if self.config.train.with_target_net:
            return "Double DQN"
        return "DQN"

    def train(self) -> None:
        tau = 5.0
        for episode in range(self.num_episodes):
            observation, _ = self.env.reset()
            terminated, truncated = False, False
            q_values = []

            episode_reward = 0.0
            while not (terminated or truncated):
                action, q_value = self.agent.act(observation, tau)
                if q_value is not None:
                    q_values.append(q_value)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += float(reward)
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
            loss_total = 0.0
            for b in range(self.config.train.update_per_batch):
                batch_experiences = self.agent.sample(self.config.train.batch_size)
                for u in range(self.config.train.update_per_batch):
                    loss = self.agent.policy_update(gamma=self.gamma, experiences=batch_experiences)
                    loss_total += loss
            episode_loss_mean = loss_total / (self.config.train.update_per_batch * self.config.train.batch_size)
            q_value_mean = np.mean(q_values)
            # decay tau
            tau = max(0.5, tau * 0.999)
            # update target net
            if episode % self.config.train.target_net_update_freq == 0:
                self.agent.polyak_update(beta=self.config.train.beta)

            print(
                f"Episode {episode}, tau: {tau}, loss: {episode_loss_mean}, "
                f"q_value_mean: {q_value_mean}, episode_reward: {episode_reward}"
            )
            if self.config.train.log_wandb:
                wandb.log(
                    {
                        "episode": episode,
                        "episode_loss_mean": episode_loss_mean,
                        "q_value_mean": q_value_mean,
                        "episode_reward": episode_reward,
                    }
                )


if __name__ == "__main__":
    simple_config = DqnConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None, solved_threshold=475.0),
        train=TrainConfig(
            num_episodes=100000,
            learning_rate=0.002,
            with_target_net=True,
            beta=0.0,
            target_net_update_freq=10,
            log_wandb=True,
        ),
    )
    trainer = DqnTrainer(simple_config)
    trainer.train()
