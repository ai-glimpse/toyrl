from dataclasses import asdict, dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


class ActorCriticNet(nn.Module):
    def __init__(self, env_dim: int, action_num: int) -> None:
        super().__init__()
        self.env_dim = env_dim
        self.action_num = action_num
        self.shared_layers = nn.Sequential(
            nn.Linear(env_dim, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, action_num)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.shared_layers(x)
        policy_action_logits = self.policy_head(x)
        v_value = self.value_head(x)
        return policy_action_logits, v_value


@dataclass
class Experience:
    observation: Any
    action: Any
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
    def __init__(self, net: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self._net = net
        self._optimizer = optimizer
        self._replay_buffer = ReplayBuffer()

    def onpolicy_reset(self) -> None:
        self._replay_buffer.reset()

    def add_experience(self, experience: Experience) -> None:
        self._replay_buffer.add_experience(experience)

    def get_buffer_total_reward(self) -> float:
        return self._replay_buffer.total_reward()

    def act(self, observation: np.ndarray, eval: bool = False) -> int:
        x = torch.from_numpy(observation.astype(np.float32))
        with torch.no_grad():
            action_logits, _ = self._net(x)
        next_action_dist = torch.distributions.Categorical(logits=action_logits)
        action = next_action_dist.sample()
        if eval:
            action = next_action_dist.probs.argmax(dim=-1)
        return action.item()

    def net_update(
        self, gamma: float, lambda_: float, value_loss_coef: float, policy_loss_coef: float, entropy_coef: float
    ) -> tuple[float, float, float]:
        experiences = self._replay_buffer.sample()

        observations = torch.tensor([exp.observation for exp in experiences])
        actions = torch.tensor([exp.action for exp in experiences])
        rewards = torch.tensor([exp.reward for exp in experiences]).unsqueeze(1)
        next_observations = torch.tensor([exp.next_observation for exp in experiences])
        terminateds = torch.tensor([exp.terminated for exp in experiences], dtype=torch.float32).unsqueeze(1)

        # calculate predicted V-values
        policy_action_logits, v_values = self._net(observations)
        # n-step return
        v_targets = torch.zeros_like(rewards)
        for t in reversed(range(len(experiences) - 1)):
            v_targets[t] = rewards[t] + gamma * v_targets[t + 1] * (1 - terminateds[t])
        # calculate value loss
        value_loss = nn.functional.mse_loss(v_values, v_targets)

        # calculate advantages by GAE
        with torch.no_grad():
            _, v_values_next = self._net(next_observations)
        deltas = rewards + gamma * v_values_next * (1 - terminateds) - v_values
        advantages = deltas.clone()
        for t in reversed(range(len(experiences) - 1)):
            advantages[t] = deltas[t] + gamma * lambda_ * advantages[t + 1] * (1 - terminateds[t])
        advantages = advantages / (advantages.std() + 1e-8)
        advantages = advantages.detach()

        action_dist = torch.distributions.Categorical(logits=policy_action_logits)
        action_entropy = action_dist.entropy().mean()
        action_log_probs = action_dist.log_prob(actions)
        # calculate policy loss
        policy_loss = -action_log_probs * advantages
        policy_loss = torch.mean(policy_loss)

        loss = value_loss * value_loss_coef + policy_loss * policy_loss_coef - entropy_coef * action_entropy

        # update
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.item(), action_entropy.item(), advantages.mean().item()


@dataclass
class EnvConfig:
    env_name: str = "CartPole-v1"
    render_mode: str | None = None
    solved_threshold: float = 475.0


@dataclass
class TrainConfig:
    gamma: float = 0.999
    lambda_: float = 0.98
    value_loss_coef: float = 0.5
    policy_loss_coef: float = 0.5
    entropy_coef: float = 0.01

    num_episodes: int = 500
    learning_rate: float = 0.01

    eval_episodes: int = 10
    eval_interval: int = 100
    log_wandb: bool = False


@dataclass
class A2CConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


class A2CTrainer:
    def __init__(self, config: A2CConfig) -> None:
        self.config = config
        self.env = gym.make(config.env.env_name, render_mode=config.env.render_mode)
        env_dim = self.env.observation_space.shape[0]  # type: ignore[index]
        action_num = self.env.action_space.n  # type: ignore[attr-defined]
        net = ActorCriticNet(env_dim=env_dim, action_num=action_num)
        optimizer = optim.Adam(net.parameters(), lr=config.train.learning_rate)
        self.agent = Agent(net=net, optimizer=optimizer)

        self.num_episodes = config.train.num_episodes
        self.gamma = config.train.gamma
        self.lambda_ = config.train.lambda_
        self.value_loss_coef = config.train.value_loss_coef
        self.policy_loss_coef = config.train.policy_loss_coef
        self.entropy_coef = config.train.entropy_coef
        self.solved_threshold = config.env.solved_threshold
        if config.train.log_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="A2C",
                name=f"[{config.env.env_name}]lr={config.train.learning_rate}",
                # track hyperparameters and run metadata
                config=asdict(config),
            )

    def train(self) -> None:
        for i, episode in enumerate(range(self.num_episodes)):
            observation, _ = self.env.reset()
            terminated, truncated = False, False
            while not (terminated or truncated):
                action = self.agent.act(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                experience = Experience(
                    observation=observation,
                    action=action,
                    reward=float(reward),
                    terminated=terminated,
                    truncated=truncated,
                    next_observation=next_observation,
                )
                self.agent.add_experience(experience)
                observation = next_observation
                if self.config.env.render_mode is not None:
                    self.env.render()
            loss, action_entropy, advantages_mean = self.agent.net_update(
                gamma=self.gamma,
                lambda_=self.lambda_,
                value_loss_coef=self.value_loss_coef,
                policy_loss_coef=self.policy_loss_coef,
                entropy_coef=self.entropy_coef,
            )
            total_reward = self.agent.get_buffer_total_reward()
            solved = total_reward > self.solved_threshold
            self.agent.onpolicy_reset()
            print(
                f"Episode {episode}, total_reward: {total_reward}, solved: {solved}, loss: {loss}, "
                f"action_entropy: {action_entropy}, advantages_mean: {advantages_mean}"
            )
            if self.config.train.log_wandb:
                wandb.log(
                    {
                        "episode": episode,
                        "loss": loss,
                        "total_reward": total_reward,
                        "action_entropy": action_entropy,
                        "advantages_mean": advantages_mean,
                    }
                )

            if i % self.config.train.eval_interval == 0:
                eval_reward = self.evaluate(self.config.train.eval_episodes)
                print(f"Episode {episode}, Eval reward: {eval_reward}")
                if self.config.train.log_wandb:
                    wandb.log({"eval_reward": eval_reward, "episode": episode})

    def evaluate(self, num_episodes: int) -> float:
        total_reward = 0.0
        for _ in range(num_episodes):
            observation, _ = self.env.reset()
            terminated, truncated = False, False
            while not (terminated or truncated):
                action = self.agent.act(observation, eval=True)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                observation = next_observation
                total_reward += float(reward)
        return total_reward / num_episodes


if __name__ == "__main__":
    default_config = A2CConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None, solved_threshold=475.0),
        train=TrainConfig(
            num_episodes=100000,
            learning_rate=7e-4,
            value_loss_coef=0.5,
            policy_loss_coef=1,
            entropy_coef=0.01,
            log_wandb=True,
        ),
    )
    trainer = A2CTrainer(default_config)
    trainer.train()
