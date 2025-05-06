"""Tests for the PPO algorithm."""

import gymnasium as gym
import numpy as np
import torch

from toyrl.ppo import (
    ActorPolicyNet,
    CriticValueNet,
    EnvConfig,
    Experience,
    PPOAgent,
    PPOConfig,
    PPOTrainer,
    ReplayBuffer,
    TrainConfig,
)


def test_actor_policy_net():
    """Test the ActorPolicyNet class."""
    env_dim, action_num = 4, 2
    net = ActorPolicyNet(env_dim=env_dim, action_num=action_num)

    # Test forward pass with single observation
    x = torch.randn(env_dim, dtype=torch.float32)
    logits = net(x)
    assert logits.shape == torch.Size([action_num])
    assert isinstance(logits, torch.Tensor)
    assert logits.dtype == torch.float32

    # Test forward pass with batched observations
    batch_size = 2
    x_batch = torch.randn(batch_size, env_dim, dtype=torch.float32)
    logits_batch = net(x_batch)
    assert logits_batch.shape == torch.Size([batch_size, action_num])
    assert isinstance(logits_batch, torch.Tensor)
    assert logits_batch.dtype == torch.float32


def test_critic_value_net():
    """Test the CriticValueNet class."""
    env_dim = 4
    net = CriticValueNet(env_dim=env_dim)

    # Test forward pass with single observation
    x = torch.randn(env_dim, dtype=torch.float32)
    value = net(x)
    assert value.shape == torch.Size([1])
    assert isinstance(value, torch.Tensor)
    assert value.dtype == torch.float32

    # Test forward pass with batched observations
    batch_size = 2
    x_batch = torch.randn(batch_size, env_dim, dtype=torch.float32)
    value_batch = net(x_batch)
    assert value_batch.shape == torch.Size([batch_size, 1])
    assert isinstance(value_batch, torch.Tensor)
    assert value_batch.dtype == torch.float32


def test_replay_buffer():
    """Test the ReplayBuffer class."""
    buffer = ReplayBuffer()

    # Test empty buffer
    assert len(buffer) == 0
    assert len(buffer.env_ids) == 0

    # Test adding experience
    exp = Experience(
        env_id=0,
        observation=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action=0,
        action_logprob=0.0,
        reward=1.0,
        next_observation=np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        terminated=False,
        truncated=False,
    )
    buffer.add_experience(exp)

    assert len(buffer) == 1
    assert len(buffer.env_ids) == 1
    assert 0 in buffer.env_ids

    # Add a second experience
    exp2 = Experience(
        env_id=1,
        observation=np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        action=1,
        action_logprob=0.0,
        reward=0.5,
        next_observation=np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32),
        terminated=False,
        truncated=False,
    )
    buffer.add_experience(exp2)

    assert len(buffer) == 2
    assert len(buffer.env_ids) == 2
    assert 1 in buffer.env_ids

    # Test sampling
    samples = buffer.sample()
    assert len(samples) == 2
    assert samples[0].action == 0
    assert samples[1].reward == 0.5

    # Test reset
    buffer.reset()
    assert len(buffer) == 0
    assert len(buffer.env_ids) == 0


def test_agent_creation():
    """Test creating a PPO agent."""
    env_dim, action_num = 4, 2
    actor = ActorPolicyNet(env_dim=env_dim, action_num=action_num)
    critic = CriticValueNet(env_dim=env_dim)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=0.01)

    agent = PPOAgent(actor=actor, critic=critic, optimizer=optimizer)

    # Check agent properties
    assert hasattr(agent, "actor")
    assert hasattr(agent, "critic")
    assert hasattr(agent, "optimizer")
    assert hasattr(agent, "_replay_buffer")


def test_agent_act():
    """Test the agent's act method."""
    env_dim, action_num = 4, 2
    actor = ActorPolicyNet(env_dim=env_dim, action_num=action_num)
    critic = CriticValueNet(env_dim=env_dim)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=0.01)

    agent = PPOAgent(actor=actor, critic=critic, optimizer=optimizer)

    # Test act method with single observation
    observation = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    action, action_logprob = agent.act(observation)

    assert isinstance(action, int)
    assert action in range(action_num)
    assert isinstance(action_logprob, float)

    # Test act method with batched observations
    batch_observation = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]], dtype=np.float32)
    action, action_logprob = agent.act(batch_observation[0])  # Test with first observation
    assert isinstance(action, int)
    assert action in range(action_num)
    assert isinstance(action_logprob, float)


def test_config():
    """Test the PPOConfig class."""
    config = PPOConfig()

    # Check default values
    assert config.env.env_name == "CartPole-v1"
    assert config.env.num_envs == 4
    assert config.train.gamma == 0.999
    assert config.train.lambda_ == 0.98
    assert config.train.epsilon == 0.2
    assert config.train.entropy_coef == 0.01
    assert config.train.time_horizons == 128
    assert config.train.update_epochs == 4
    assert config.train.num_minibatches == 4

    # Test custom config
    custom_config = PPOConfig(
        env=EnvConfig(env_name="MountainCar-v0", solved_threshold=90.0),
        train=TrainConfig(
            gamma=0.99,
            lambda_=0.95,
            epsilon=0.1,
            entropy_coef=0.02,
            time_horizons=64,
            update_epochs=8,
            num_minibatches=8,
            learning_rate=0.005,
        ),
    )

    assert custom_config.env.env_name == "MountainCar-v0"
    assert custom_config.env.solved_threshold == 90.0
    assert custom_config.train.gamma == 0.99
    assert custom_config.train.lambda_ == 0.95
    assert custom_config.train.epsilon == 0.1
    assert custom_config.train.entropy_coef == 0.02
    assert custom_config.train.time_horizons == 64
    assert custom_config.train.update_epochs == 8
    assert custom_config.train.num_minibatches == 8
    assert custom_config.train.learning_rate == 0.005


def test_trainer_creation():
    """Test creating a PPO trainer."""
    config = PPOConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None),
        train=TrainConfig(total_timesteps=1000, learning_rate=0.01, log_wandb=False),
    )

    trainer = PPOTrainer(config)

    assert isinstance(trainer.envs, gym.vector.VectorEnv)
    assert isinstance(trainer.agent, PPOAgent)
    assert hasattr(trainer, "config")
    assert trainer.config == config


def test_agent_net_update():
    """Test the agent's net_update method."""
    env_dim, action_num = 4, 2
    actor = ActorPolicyNet(env_dim=env_dim, action_num=action_num)
    critic = CriticValueNet(env_dim=env_dim)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=0.01)

    agent = PPOAgent(actor=actor, critic=critic, optimizer=optimizer)

    # Add some experiences
    exp1 = Experience(
        env_id=0,
        observation=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action=0,
        action_logprob=0.0,
        reward=1.0,
        next_observation=np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        terminated=False,
        truncated=False,
    )
    agent.add_experience(exp1)

    exp2 = Experience(
        env_id=0,
        observation=np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        action=1,
        action_logprob=0.0,
        reward=0.5,
        next_observation=np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32),
        terminated=True,
        truncated=False,
    )
    agent.add_experience(exp2)

    # Test net_update
    loss = agent.net_update(
        num_minibatches=1,
        gamma=0.99,
        lambda_=0.95,
        epsilon=0.2,
        entropy_coef=0.01,
    )

    assert isinstance(loss, float)

    # Check that buffer is still intact
    assert len(agent._replay_buffer) == 2


def test_minimal_training():
    """Test minimal training run with a small number of timesteps."""
    # Create minimal config with just a few timesteps
    config = PPOConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None, num_envs=2),
        train=TrainConfig(
            total_timesteps=256,  # 2 environments * 128 timesteps
            time_horizons=128,
            learning_rate=0.01,
            log_wandb=False,
            update_epochs=1,  # Reduce update epochs for faster testing
            num_minibatches=1,  # Reduce minibatches for faster testing
        ),
    )

    # Initialize trainer
    trainer = PPOTrainer(config)

    # Run training
    trainer.train()

    # If we got here without errors, test passed
    assert True
