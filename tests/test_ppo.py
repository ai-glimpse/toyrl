"""Tests for the PPO algorithm."""

import gymnasium as gym
import torch

from toyrl.ppo import (
    ActorPolicyNet,
    CriticValueNet,
    PPOAgent,
    PPOConfig,
    PPOTrainer,
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


def test_config():
    """Test the PPOConfig class."""
    config = PPOConfig()

    # Check default values
    assert config.env_name == "CartPole-v1"
    assert config.num_envs == 4
    assert config.gamma == 0.999
    assert config.lambda_ == 0.98
    assert config.epsilon == 0.2
    assert config.entropy_coef == 0.01
    assert config.time_horizons == 128
    assert config.update_epochs == 4
    assert config.num_minibatches == 4

    # Test custom config
    custom_config = PPOConfig(
        env_name="MountainCar-v0",
        solved_threshold=90.0,
        gamma=0.99,
        lambda_=0.95,
        epsilon=0.1,
        entropy_coef=0.02,
        time_horizons=64,
        update_epochs=8,
        num_minibatches=8,
        learning_rate=0.005,
    )

    assert custom_config.env_name == "MountainCar-v0"
    assert custom_config.solved_threshold == 90.0
    assert custom_config.gamma == 0.99
    assert custom_config.lambda_ == 0.95
    assert custom_config.epsilon == 0.1
    assert custom_config.entropy_coef == 0.02
    assert custom_config.time_horizons == 64
    assert custom_config.update_epochs == 8
    assert custom_config.num_minibatches == 8
    assert custom_config.learning_rate == 0.005


def test_trainer_creation():
    """Test creating a PPO trainer."""
    config = PPOConfig(
        env_name="CartPole-v1",
        render_mode=None,
        total_timesteps=1000,
        learning_rate=0.01,
        log_wandb=False,
    )

    trainer = PPOTrainer(config)

    assert isinstance(trainer.envs, gym.vector.VectorEnv)
    assert isinstance(trainer.agent, PPOAgent)
    assert hasattr(trainer, "config")
    assert trainer.config == config


def test_minimal_training():
    """Test minimal training run with a small number of timesteps."""
    # Create minimal config with just a few timesteps
    config = PPOConfig(
        env_name="CartPole-v1",
        render_mode=None,
        num_envs=2,
        total_timesteps=256,  # 2 environments * 128 timesteps
        time_horizons=128,
        learning_rate=0.01,
        log_wandb=False,
        update_epochs=1,  # Reduce update epochs for faster testing
        num_minibatches=1,  # Reduce minibatches for faster testing
    )

    # Initialize trainer
    trainer = PPOTrainer(config)

    # Run training
    trainer.train()

    # If we got here without errors, test passed
    assert True
