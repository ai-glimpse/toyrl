import random
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
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_num),
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
    replay_buffer_size: int = 10000
    buffer: list[Experience] = field(default_factory=list)
    _head_pointer: int = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add_experience(self, experience: Experience) -> None:
        if len(self.buffer) < self.replay_buffer_size:
            # Buffer not full yet, append new experience
            self.buffer.append(experience)
        else:
            # Buffer full, overwrite oldest experience
            index = self._head_pointer % self.replay_buffer_size
            self.buffer[index] = experience

        # Increment pointer
        self._head_pointer += 1

    def reset(self) -> None:
        self.buffer = []
        self._head_pointer = 0

    def sample(self, batch_size: int) -> list[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))


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

    def act(self, observation: np.floating, tau: float) -> tuple[int, float]:
        x = torch.from_numpy(observation.astype(np.float32))
        with torch.no_grad():
            logits = self._policy_net(x)
        next_action = torch.distributions.Categorical(logits=logits / tau).sample().item()
        q_value = logits[next_action].item()
        return next_action, q_value

    def sample(self, batch_size: int) -> list[Experience]:
        return self._replay_buffer.sample(batch_size)

    def policy_update(self, gamma: float, experiences: list[Experience]) -> float:
        observations = torch.tensor([experience.observation for experience in experiences])
        actions = torch.tensor([experience.action for experience in experiences], dtype=torch.float32)
        next_observations = torch.tensor([experience.next_observation for experience in experiences])
        rewards = torch.tensor([experience.reward for experience in experiences])
        terminated = torch.tensor(
            [experience.terminated for experience in experiences],
            dtype=torch.float32,
        )

        # q preds
        action_q_preds = self._policy_net(observations).gather(1, actions.long().unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_action_logits = self._policy_net(next_observations)
            next_actions = torch.argmax(next_action_logits, dim=1)
            if self._target_net is None:  # Vanilla DQN
                next_action_q_preds = torch.gather(next_action_logits, 1, next_actions.unsqueeze(1)).squeeze(1)
            else:  # Double DQN
                next_action_q_preds = (
                    self._target_net(next_observations).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                )
        action_q_targets = rewards + gamma * (1 - terminated) * next_action_q_preds
        loss = torch.nn.functional.mse_loss(action_q_preds, action_q_targets)
        # update
        self._optimizer.zero_grad()
        loss.backward()
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
    replay_buffer_capacity: int = 10000
    """The maximum capacity of the experience replay buffer."""

    max_training_steps: int = 500000
    """The maximum number of environment steps to train for."""
    learning_starts: int = 10000
    """The number of steps to collect before starting learning."""
    policy_update_frequency: int = 10
    """How often to update the policy network (in environment steps)."""
    batches_per_training_step: int = 16
    """The number of experience batches to sample in each training step."""
    batch_size: int = 128
    """The size of each training batch."""
    updates_per_batch: int = 1
    """The number of optimization steps to perform on each batch."""

    learning_rate: float = 0.01
    """The learning rate for the optimizer."""

    use_target_network: bool = False
    """Whether to use a separate target network (Double DQN when True)."""
    target_update_frequency: int = 10
    """How often to update the target network (in environment steps)."""
    target_soft_update_beta: float = 0.0
    """The soft update parameter for target network (0.0 means hard update)."""

    log_wandb: bool = False
    """Whether to log the training process to Weights and Biases."""


@dataclass
class DqnConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


class DqnTrainer:
    def __init__(self, config: DqnConfig) -> None:
        self.config = config
        self.env = self._make_env()
        if isinstance(self.env.action_space, gym.spaces.Discrete) is False:
            raise ValueError("Only discrete action space is supported.")
        env_dim = self.env.observation_space.shape[0]  # type: ignore[index]
        action_num = self.env.action_space.n  # type: ignore[attr-defined]

        policy_net = PolicyNet(env_dim=env_dim, action_num=action_num)
        optimizer = optim.Adam(policy_net.parameters(), lr=config.train.learning_rate)
        if config.train.use_target_network:
            target_net = PolicyNet(env_dim=env_dim, action_num=action_num)
            target_net.load_state_dict(policy_net.state_dict())
        else:
            target_net = None
        self.agent = Agent(
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            replay_buffer_size=config.train.replay_buffer_capacity,
        )

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

    def _make_env(self):
        env = gym.make(self.config.env.env_name, render_mode=self.config.env.render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.Autoreset(env)
        return env

    def _get_dqn_name(self) -> str:
        if self.config.train.use_target_network:
            return "Double DQN"
        return "DQN"

    def train(self) -> None:
        tau = 5.0
        global_step = 0
        observation, _ = self.env.reset()
        while global_step < self.config.train.max_training_steps:
            global_step += 1
            # decay tau
            tau = max(0.1, tau * 0.995)

            action, q_value = self.agent.act(observation, tau)
            if self.config.train.log_wandb:
                wandb.log({"global_step": global_step, "q_value": q_value})

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

            if terminated or truncated:
                if info and "episode" in info:
                    reward = info["episode"]["r"]
                    print(f"global_step={global_step}, episodic_return={reward}")
                    if self.config.train.log_wandb:
                        wandb.log(
                            {
                                "global_step": global_step,
                                "episode_reward": reward,
                            }
                        )

            if self.env.render_mode is not None:
                self.env.render()

            if (
                global_step >= self.config.train.learning_starts
                and global_step % self.config.train.policy_update_frequency == 0
            ):
                loss = self._train_step()
                if self.config.train.log_wandb:
                    wandb.log(
                        {
                            "global_step": global_step,
                            "loss": loss,
                        }
                    )
            # update target net
            if self.config.train.use_target_network and global_step % self.config.train.target_update_frequency == 0:
                self.agent.polyak_update(beta=self.config.train.target_soft_update_beta)

    def _train_step(self) -> float:
        loss = 0.0
        for b in range(self.config.train.batches_per_training_step):
            batch_experiences = self.agent.sample(self.config.train.batch_size)
            for u in range(self.config.train.updates_per_batch):
                loss += self.agent.policy_update(gamma=self.gamma, experiences=batch_experiences)
        loss /= self.config.train.batches_per_training_step * self.config.train.updates_per_batch
        return loss


if __name__ == "__main__":
    simple_config = DqnConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None, solved_threshold=475.0),
        train=TrainConfig(
            max_training_steps=500000,
            learning_rate=2.5e-4,
            use_target_network=True,
            target_soft_update_beta=0.0,
            target_update_frequency=5,
            log_wandb=True,
        ),
    )
    trainer = DqnTrainer(simple_config)
    trainer.train()
