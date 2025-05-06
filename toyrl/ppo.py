from dataclasses import asdict, dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


class ActorPolicyNet(nn.Module):
    def __init__(self, env_dim: int, action_num: int) -> None:
        super().__init__()
        layers = [
            nn.Linear(env_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
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
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # type: ignore


@dataclass
class Experience:
    env_id: int
    terminated: bool
    truncated: bool
    observation: Any
    reward: float
    next_observation: Any

    action: Any
    action_logprob: float
    advantage: float | None = None
    target_value: float | None = None


@dataclass
class ReplayBuffer:
    buffer: list[Experience] = field(default_factory=list)
    env_ids: set[int] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.buffer)

    def add_experience(self, experience: Experience) -> None:
        self.buffer.append(experience)
        self.env_ids.add(experience.env_id)

    def reset(self) -> None:
        self.buffer = []
        self.env_ids = set()

    def sample(self) -> list[Experience]:
        return self.buffer


@dataclass
class EnvConfig:
    env_name: str = "CartPole-v1"
    num_envs: int = 4
    """The number of parallel game environments"""
    render_mode: str | None = None
    solved_threshold: float = 475.0


@dataclass
class TrainConfig:
    gamma: float = 0.999
    lambda_: float = 0.98
    epsilon: float = 0.2
    entropy_coef: float = 0.01

    total_timesteps: int = 500000
    time_horizons: int = 128  # T
    """The number of time steps to collect before updating the policy"""
    update_epochs: int = 4  # K
    """The K epochs to update the policy"""
    num_minibatches: int = 4
    """The number of mini-batches"""

    learning_rate: float = 2.5e-4
    anneal_learning_rate: bool = True
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
        self._replay_buffer = ReplayBuffer()

    def add_experience(self, experience: Experience) -> None:
        self._replay_buffer.add_experience(experience)

    def reset(self) -> None:
        self._replay_buffer.reset()

    def act(self, observation: Any) -> tuple[int, float]:
        x = torch.from_numpy(observation.astype(np.float32))
        with torch.no_grad():
            logits = self.actor(x)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        action = torch.distributions.Categorical(probs).sample()
        action_logprob = torch.log(probs[action])
        return action.item(), action_logprob.item()

    def net_update(
        self,
        num_minibatches: int,
        gamma: float,
        lambda_: float,
        epsilon: float,
        entropy_coef: float,
    ) -> float:
        raw_experiences = self._replay_buffer.sample()
        # calculate advantages and target values by GAE
        experiences = self._calc_adv_v_target(raw_experiences, gamma, lambda_)
        minibatch_size = len(experiences) // num_minibatches
        total_loss = 0.0
        for i in range(num_minibatches):
            batch_experiences = experiences[minibatch_size * i : minibatch_size * (i + 1)]
            observations = torch.tensor(np.array([exp.observation for exp in batch_experiences]), dtype=torch.float32)
            actions = torch.tensor(np.array([exp.action for exp in batch_experiences]), dtype=torch.int64)
            old_action_logprobs = torch.tensor(
                np.array([exp.action_logprob for exp in batch_experiences]), dtype=torch.float32
            )
            advantages = torch.tensor(np.array([exp.advantage for exp in batch_experiences]), dtype=torch.float32)
            target_v_values = torch.tensor(
                np.array([exp.target_value for exp in batch_experiences]), dtype=torch.float32
            )

            # critic value loss
            v_values = self.critic(observations).squeeze(1)
            critic_value_loss = torch.nn.functional.mse_loss(v_values, target_v_values)

            # actor policy loss
            action_logits = self.actor(observations)
            action_probs = torch.nn.functional.softmax(action_logits, dim=-1)
            action_entropy = torch.distributions.Categorical(action_probs).entropy()
            action_logprobs = torch.gather(action_probs.log(), 1, actions.unsqueeze(1)).squeeze(1)
            ratios = torch.exp(action_logprobs - old_action_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
            actor_policy_loss = -torch.min(surr1, surr2).mean() - entropy_coef * action_entropy.mean()

            loss = actor_policy_loss + critic_value_loss
            # update actor and critic
            self.optimizer.zero_grad()
            loss.backward()
            # clip
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / num_minibatches

    def _calc_adv_v_target(self, experiences: list[Experience], gamma: float, lambda_: float) -> list[Experience]:
        """calculate advantages and target values by GAE for each env_id"""
        for env_id in self._replay_buffer.env_ids:
            env_experiences = [exp for exp in experiences if exp.env_id == env_id]
            rewards = torch.tensor([exp.reward for exp in env_experiences], dtype=torch.float32)
            terminateds = torch.tensor([exp.terminated for exp in env_experiences], dtype=torch.float32)
            with torch.no_grad():
                values = self.critic(
                    torch.tensor(np.array([exp.observation for exp in env_experiences]), dtype=torch.float32)
                ).squeeze(1)
                next_values = self.critic(
                    torch.tensor(np.array([exp.next_observation for exp in env_experiences]), dtype=torch.float32)
                ).squeeze(1)
            deltas = rewards + gamma * (1 - terminateds) * next_values - values
            advantages = torch.empty_like(deltas).detach()
            for t in reversed(range(len(deltas))):
                if t == len(deltas) - 1:
                    advantages[t] = deltas[t]
                else:
                    advantages[t] = deltas[t] + gamma * lambda_ * (1 - terminateds[t]) * advantages[t + 1]
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            target_values = advantages + values
            for i, exp in enumerate(env_experiences):
                exp.advantage = advantages[i].item()
                exp.target_value = target_values[i].item()
        return experiences


class PPOTrainer:
    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.envs = self._make_env()
        env_dim = self.envs.single_observation_space.shape[0]  # type: ignore[index]
        action_num = self.envs.single_action_space.n  # type: ignore[attr-defined]
        actor = ActorPolicyNet(env_dim=env_dim, action_num=action_num)
        critic = CriticValueNet(env_dim=env_dim)
        optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=config.train.learning_rate
        )
        self.agent = PPOAgent(actor=actor, critic=critic, optimizer=optimizer)
        if config.train.log_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="PPO",
                name=f"[{config.env.env_name}]lr={config.train.learning_rate}",
                # track hyperparameters and run metadata
                config=asdict(config),
            )

    def _make_env(self):
        envs = gym.make_vec(
            id=self.config.env.env_name,
            num_envs=self.config.env.num_envs,
            render_mode=self.config.env.render_mode,
        )
        envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
        return envs

    def train(self):
        batch_size = self.config.train.time_horizons * self.config.env.num_envs
        num_iteration = self.config.train.total_timesteps // batch_size

        global_step = 0
        observations, _ = self.envs.reset()
        for iteration in range(num_iteration):
            if self.config.train.anneal_learning_rate:
                frac = 1.0 - iteration / num_iteration
                lr = frac * self.config.train.learning_rate
                self.agent.optimizer.param_groups[0]["lr"] = lr

            # Collect experience
            for step in range(self.config.train.time_horizons):
                global_step += self.config.env.num_envs
                actions, action_logprobs = [], []
                for obs in observations:
                    action, action_logprob = self.agent.act(obs)
                    actions.append(action)
                    action_logprobs.append(action_logprob)
                next_observations, rewards, terminateds, truncateds, infos = self.envs.step(np.array(actions))
                for env_id in range(self.config.env.num_envs):
                    experience = Experience(
                        env_id=env_id,
                        terminated=terminateds[env_id],
                        truncated=truncateds[env_id],
                        observation=observations[env_id],
                        action=actions[env_id],
                        action_logprob=action_logprobs[env_id],
                        reward=float(rewards[env_id]),
                        next_observation=next_observations[env_id],
                    )
                    self.agent.add_experience(experience)
                observations = next_observations

                if "episode" in infos:
                    for i in range(self.config.env.num_envs):
                        if infos["_episode"][i]:
                            print(f"global_step={global_step}, episodic_return={infos['episode']['r'][i]}")
                            if self.config.train.log_wandb:
                                wandb.log(
                                    {
                                        "global_step": global_step,
                                        "episodic_return": infos["episode"]["r"][i],
                                    }
                                )

            # Update policy
            total_loss = 0.0
            for _ in range(self.config.train.update_epochs):
                loss = self.agent.net_update(
                    gamma=self.config.train.gamma,
                    lambda_=self.config.train.lambda_,
                    epsilon=self.config.train.epsilon,
                    entropy_coef=self.config.train.entropy_coef,
                    num_minibatches=self.config.train.num_minibatches,
                )
                total_loss += loss
            loss = total_loss / self.config.train.update_epochs
            if self.config.train.log_wandb:
                wandb.log(
                    {
                        "global_step": global_step,
                        "learning_rate": self.agent.optimizer.param_groups[0]["lr"],
                        "loss": loss,
                    }
                )
            # Onpolicy reset
            self.agent.reset()


if __name__ == "__main__":
    default_config = PPOConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None, solved_threshold=475.0),
        train=TrainConfig(
            gamma=0.99,
            lambda_=0.95,
            epsilon=0.2,
            entropy_coef=0.01,
            total_timesteps=1000000,
            time_horizons=256,
            update_epochs=4,
            num_minibatches=4,
            learning_rate=2.5e-4,
            log_wandb=True,
        ),
    )
    trainer = PPOTrainer(default_config)
    trainer.train()
