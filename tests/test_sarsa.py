"""Tests for the SARSA algorithm."""

import gymnasium as gym
import numpy as np
import torch

from toyrl.sarsa import (
    Agent,
    EnvConfig,
    Experience,
    PolicyNet,
    ReplayBuffer,
    SarsaConfig,
    SarsaTrainer,
    TrainConfig,
)


def test_policy_net():
    """Test the PolicyNet class."""
    env_dim, action_num = 4, 2
    net = PolicyNet(env_dim=env_dim, action_num=action_num)

    # Test forward pass
    x = torch.randn(env_dim, dtype=torch.float32)
    output = net(x)

    assert output.shape == torch.Size([action_num])
    assert isinstance(output, torch.Tensor)
    assert output.dtype == torch.float32


def test_replay_buffer():
    """Test the ReplayBuffer class."""
    buffer = ReplayBuffer()

    # Test empty buffer
    assert len(buffer) == 0

    # Test adding experience
    exp = Experience(
        observation=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action=0,
        reward=1.0,
        next_observation=np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        next_action=1,
        terminated=False,
        truncated=False,
    )
    buffer.add_experience(exp)

    assert len(buffer) == 1

    # Add a second experience
    exp2 = Experience(
        observation=np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        action=1,
        reward=0.5,
        next_observation=np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32),
        terminated=False,
        truncated=False,
    )
    buffer.add_experience(exp2)

    # Test sampling with next state-action pairs
    samples = buffer.sample(with_next_sa=True)
    assert len(samples) == 1
    assert samples[0].action == 0
    assert samples[0].next_action == 1

    # Test reset
    buffer.reset()
    assert len(buffer) == 0


def test_agent_creation():
    """Test creating an agent."""
    env_dim, action_num = 4, 2
    policy_net = PolicyNet(env_dim=env_dim, action_num=action_num)
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=0.01)

    agent = Agent(policy_net=policy_net, optimizer=optimizer)

    # Check agent properties
    assert hasattr(agent, "_policy_net")
    assert hasattr(agent, "_optimizer")
    assert hasattr(agent, "_replay_buffer")
    assert agent._action_num == action_num


def test_agent_act():
    """Test the agent's act method."""
    env_dim, action_num = 4, 2
    policy_net = PolicyNet(env_dim=env_dim, action_num=action_num)
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=0.01)

    agent = Agent(policy_net=policy_net, optimizer=optimizer)

    # Test act method with epsilon=0 (greedy)
    observation = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    action = agent.act(observation, epsilon=0.0)

    assert isinstance(action, int)
    assert action in [0, 1]  # For CartPole

    # Test act method with epsilon=1.0 (random)
    action = agent.act(observation, epsilon=1.0)
    assert action in [0, 1]


def test_config():
    """Test the Config class."""
    config = SarsaConfig()

    # Check default values
    assert config.env.env_name == "CartPole-v1"
    assert config.train.gamma == 0.999

    # Test custom config
    custom_config = SarsaConfig(
        env=EnvConfig(env_name="MountainCar-v0", solved_threshold=90.0),
        train=TrainConfig(max_training_steps=1000, learning_rate=0.005),
    )

    assert custom_config.env.env_name == "MountainCar-v0"
    assert custom_config.env.solved_threshold == 90.0
    assert custom_config.train.max_training_steps == 1000
    assert custom_config.train.learning_rate == 0.005


def test_trainer_creation():
    """Test creating a trainer."""
    config = SarsaConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None),
        train=TrainConfig(max_training_steps=10, learning_rate=0.01, log_wandb=False),
    )

    trainer = SarsaTrainer(config)

    assert isinstance(trainer.env, gym.Env)
    assert isinstance(trainer.agent, Agent)
    assert hasattr(trainer, "gamma")
    assert trainer.gamma == 0.999


def test_minimal_training():
    """Test minimal training run with a single episode."""
    # Create minimal config with just one step
    config = SarsaConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None),
        train=TrainConfig(max_training_steps=100, learning_rate=0.01, log_wandb=False),
    )

    # Initialize trainer
    trainer = SarsaTrainer(config)

    # Run training for one step
    trainer.train()

    # If we got here without errors, test passed
    assert True
