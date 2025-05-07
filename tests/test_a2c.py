"""Tests for the A2C algorithm."""

import gymnasium as gym
import numpy as np
import torch

from toyrl.a2c import (
    A2CConfig,
    A2CTrainer,
    ActorCriticNet,
    Agent,
    Experience,
    ReplayBuffer,
)


def test_actor_critic_net():
    """Test the ActorCriticNet class."""
    in_dim, out_dim = 4, 2
    net = ActorCriticNet(env_dim=in_dim, action_num=out_dim)

    # Test forward pass
    x = torch.randn(in_dim)
    policy_logits, value = net(x)

    assert policy_logits.shape == torch.Size([out_dim])
    assert value.shape == torch.Size([1])
    assert isinstance(policy_logits, torch.Tensor)
    assert isinstance(value, torch.Tensor)


def test_replay_buffer():
    """Test the ReplayBuffer class."""
    buffer = ReplayBuffer()

    # Test empty buffer
    assert len(buffer) == 0
    assert buffer.total_reward() == 0

    # Test adding experience
    exp = Experience(
        observation=np.array([0.0, 0.0, 0.0, 0.0]),
        action=0,
        reward=1.0,
        next_observation=np.array([0.1, 0.1, 0.1, 0.1]),
        terminated=False,
        truncated=False,
    )
    buffer.add_experience(exp)

    assert len(buffer) == 1
    assert buffer.total_reward() == 1.0

    # Test reset
    buffer.reset()
    assert len(buffer) == 0


def test_agent_creation():
    """Test creating an agent."""
    in_dim, out_dim = 4, 2
    net = ActorCriticNet(env_dim=in_dim, action_num=out_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    agent = Agent(net=net, optimizer=optimizer)

    # Check agent properties
    assert hasattr(agent, "_net")
    assert hasattr(agent, "_optimizer")
    assert hasattr(agent, "_replay_buffer")


def test_agent_act():
    """Test the agent's act method."""
    in_dim, out_dim = 4, 2
    net = ActorCriticNet(env_dim=in_dim, action_num=out_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    agent = Agent(net=net, optimizer=optimizer)

    # Test act method
    observation = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    action = agent.act(observation)

    assert isinstance(action, int)
    assert action in [0, 1]  # For CartPole


def test_config():
    """Test the Config class."""
    config = A2CConfig()

    # Check default values
    assert config.env_name == "CartPole-v1"
    assert config.gamma == 0.999

    # Test custom config
    custom_config = A2CConfig(
        env_name="MountainCar-v0",
        solved_threshold=90.0,
        num_episodes=1000,
        learning_rate=0.005,
    )

    assert custom_config.env_name == "MountainCar-v0"
    assert custom_config.solved_threshold == 90.0
    assert custom_config.num_episodes == 1000
    assert custom_config.learning_rate == 0.005


def test_trainer_creation():
    """Test creating a trainer."""
    config = A2CConfig(
        env_name="CartPole-v1",
        render_mode=None,
        num_episodes=10,
        learning_rate=0.01,
        log_wandb=False,
    )

    trainer = A2CTrainer(config)

    assert isinstance(trainer.env, gym.Env)
    assert isinstance(trainer.agent, Agent)
    assert hasattr(trainer, "num_episodes")
    assert trainer.num_episodes == 10


def test_minimal_training():
    """Test minimal training run with a single episode."""
    # Create minimal config with just one episode
    config = A2CConfig(
        env_name="CartPole-v1",
        render_mode=None,
        num_episodes=1,
        learning_rate=0.01,
        log_wandb=False,
    )

    # Initialize trainer
    trainer = A2CTrainer(config)

    # Run training for one episode
    trainer.train()

    # If we got here without errors, test passed
    assert True
