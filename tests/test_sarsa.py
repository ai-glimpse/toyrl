"""Tests for the SARSA algorithm."""

import numpy as np
import torch

from toyrl.sarsa import (
    Agent,
    Experience,
    PolicyNet,
    ReplayBuffer,
    SarsaConfig,
    SarsaTrainer,
)


def test_policy_net():
    """Test the PolicyNet class."""
    in_dim = 4
    out_dim = 2
    net = PolicyNet(env_dim=in_dim, action_num=out_dim)
    x = torch.randn(in_dim)
    output = net(x)
    assert output.shape == torch.Size([out_dim])
    assert isinstance(output, torch.Tensor)


def test_replay_buffer():
    """Test the ReplayBuffer class."""
    buffer = ReplayBuffer()
    assert len(buffer) == 0

    # Test adding experience
    exp = Experience(
        observation=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action=0,
        reward=1.0,
        next_observation=np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        terminated=False,
        truncated=False,
    )
    buffer.add_experience(exp)
    assert len(buffer) == 1

    # Test sampling with next state-action pairs
    samples = buffer.sample(with_next_sa=True)
    assert len(samples) == 0  # No next state-action pairs yet

    # Add a second experience to test next state-action pairs
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


def test_agent():
    """Test creating an agent."""
    in_dim = 4
    out_dim = 2
    net = PolicyNet(env_dim=in_dim, action_num=out_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    agent = Agent(policy_net=net, optimizer=optimizer)

    # Test act method with epsilon=0 (greedy)
    state = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    action = agent.act(state, epsilon=0.0)
    assert isinstance(action, int)
    assert 0 <= action < out_dim

    # Test act method with epsilon=1.0 (random)
    action = agent.act(state, epsilon=1.0)
    assert isinstance(action, int)
    assert 0 <= action < out_dim


def test_config():
    """Test the Config class."""
    config = SarsaConfig()
    assert config.env_name == "CartPole-v1"
    assert config.render_mode is None
    assert config.solved_threshold == 475.0
    assert config.gamma == 0.999
    assert config.max_training_steps == 500000
    assert config.learning_rate == 2.5e-4
    assert config.log_wandb is False


def test_trainer():
    """Test creating a trainer."""
    config = SarsaConfig(
        env_name="CartPole-v1",
        render_mode=None,
        solved_threshold=475.0,
        max_training_steps=1000,
        learning_rate=0.01,
        log_wandb=False,
    )
    trainer = SarsaTrainer(config)
    assert trainer.config == config
    assert trainer.gamma == config.gamma
    assert trainer.solved_threshold == config.solved_threshold


def test_minimal_training():
    """Test minimal training run with a single episode."""
    # Create minimal config with just one step
    config = SarsaConfig(
        env_name="CartPole-v1",
        render_mode=None,
        solved_threshold=475.0,
        max_training_steps=100,
        learning_rate=0.01,
        log_wandb=False,
    )

    # Initialize trainer
    trainer = SarsaTrainer(config)

    # Run training for one step
    trainer.train()

    # If we got here without errors, test passed
    assert True
