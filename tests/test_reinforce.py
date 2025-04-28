"""Tests for the reinforce algorithm."""

import gymnasium as gym
import numpy as np
import torch

from toyrl.reinforce import (
    Agent,
    EnvConfig,
    Experience,
    PolicyNet,
    ReinforceConfig,
    ReinforceTrainer,
    ReplayBuffer,
    TrainConfig,
)


def test_policy_net():
    """Test the PolicyNet class."""
    in_dim, out_dim = 4, 2
    net = PolicyNet(env_dim=in_dim, action_num=out_dim)

    # Test forward pass
    x = torch.randn(in_dim)
    output = net(x)

    assert output.shape == torch.Size([out_dim])
    assert isinstance(output, torch.Tensor)


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
        action_log_prob=torch.tensor(0.5),
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
    policy_net = PolicyNet(env_dim=in_dim, action_num=out_dim)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.01)

    agent = Agent(policy_net=policy_net, optimizer=optimizer)

    # Check agent properties
    assert hasattr(agent, "_policy_net")
    assert hasattr(agent, "_optimizer")
    assert hasattr(agent, "_replay_buffer")


def test_agent_act():
    """Test the agent's act method."""
    in_dim, out_dim = 4, 2
    policy_net = PolicyNet(env_dim=in_dim, action_num=out_dim)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.01)

    agent = Agent(policy_net=policy_net, optimizer=optimizer)

    # Test act method
    observation = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    action, action_log_prob = agent.act(observation)

    assert isinstance(action, int)
    assert action in [0, 1]  # For CartPole
    assert isinstance(action_log_prob, torch.Tensor)


def test_config():
    """Test the Config class."""
    config = ReinforceConfig()

    # Check default values
    assert config.env.env_name == "CartPole-v1"
    assert config.train.gamma == 0.999

    # Test custom config
    custom_config = ReinforceConfig(
        env=EnvConfig(env_name="MountainCar-v0", solved_threshold=90.0),
        train=TrainConfig(num_episodes=1000, learning_rate=0.005),
    )

    assert custom_config.env.env_name == "MountainCar-v0"
    assert custom_config.env.solved_threshold == 90.0
    assert custom_config.train.num_episodes == 1000
    assert custom_config.train.learning_rate == 0.005


def test_trainer_creation():
    """Test creating a trainer."""
    config = ReinforceConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None),
        train=TrainConfig(num_episodes=10, learning_rate=0.01, log_wandb=False),
    )

    trainer = ReinforceTrainer(config)

    assert isinstance(trainer.env, gym.Env)
    assert isinstance(trainer.agent, Agent)
    assert hasattr(trainer, "num_episodes")
    assert trainer.num_episodes == 10


def test_minimal_training():
    """Test minimal training run with a single episode."""
    # Create minimal config with just one episode
    config = ReinforceConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None),
        train=TrainConfig(num_episodes=1, learning_rate=0.01, log_wandb=False),
    )

    # Initialize trainer
    trainer = ReinforceTrainer(config)

    # Run training for one episode
    trainer.train()

    # If we got here without errors, test passed
    assert True
