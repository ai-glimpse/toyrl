"""Tests for the DQN algorithm."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from toyrl.dqn import (
    Agent,
    DqnConfig,
    DqnTrainer,
    EnvConfig,
    Experience,
    PolicyNet,
    ReplayBuffer,
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
    buffer = ReplayBuffer(replay_buffer_size=1000)

    # Test empty buffer
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

    # Test sampling
    samples = buffer.sample(batch_size=1)
    assert len(samples) == 1
    assert samples[0].action == 0
    assert samples[0].reward == 1.0

    # Test buffer size limit
    for _ in range(2000):  # Add more experiences than buffer size
        buffer.add_experience(exp)
    assert len(buffer) <= 1000  # Should not exceed buffer size

    # Test reset
    buffer.reset()
    assert len(buffer) == 0


def test_agent_creation():
    """Test creating an agent."""
    env_dim, action_num = 4, 2
    policy_net = PolicyNet(env_dim=env_dim, action_num=action_num)
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=0.01)

    # Test with target network
    target_net = PolicyNet(env_dim=env_dim, action_num=action_num)
    agent = Agent(
        policy_net=policy_net,
        target_net=target_net,
        optimizer=optimizer,
        replay_buffer_size=1000,
    )

    # Check agent properties
    assert hasattr(agent, "_policy_net")
    assert hasattr(agent, "_target_net")
    assert hasattr(agent, "_optimizer")
    assert hasattr(agent, "_replay_buffer")
    assert agent._action_num == action_num

    # Test without target network
    agent = Agent(
        policy_net=policy_net,
        target_net=None,
        optimizer=optimizer,
        replay_buffer_size=1000,
    )
    assert agent._target_net is None


def test_agent_act():
    """Test the agent's act method."""
    env_dim, action_num = 4, 2
    policy_net = PolicyNet(env_dim=env_dim, action_num=action_num)
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=0.01)

    agent = Agent(
        policy_net=policy_net,
        target_net=None,
        optimizer=optimizer,
        replay_buffer_size=1000,
    )

    # Test act method
    observation = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    action, q_value = agent.act(observation, tau=1.0)

    assert isinstance(action, int)
    assert action in [0, 1]  # For CartPole
    assert isinstance(q_value, float)


def test_agent_policy_update():
    """Test the agent's policy update method."""
    env_dim, action_num = 4, 2
    policy_net = PolicyNet(env_dim=env_dim, action_num=action_num)
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=0.01)

    # Test with target network
    target_net = PolicyNet(env_dim=env_dim, action_num=action_num)
    agent = Agent(
        policy_net=policy_net,
        target_net=target_net,
        optimizer=optimizer,
        replay_buffer_size=1000,
    )

    # Create some experiences
    experiences = [
        Experience(
            observation=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            action=0,
            reward=1.0,
            next_observation=np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
            terminated=False,
            truncated=False,
        ),
        Experience(
            observation=np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
            action=1,
            reward=0.5,
            next_observation=np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32),
            terminated=False,
            truncated=False,
        ),
    ]

    # Test policy update
    loss = agent.policy_update(gamma=0.99, experiences=experiences)
    assert isinstance(loss, float)
    assert loss >= 0.0  # Loss should be non-negative


def test_agent_polyak_update():
    """Test the agent's polyak update method."""
    env_dim, action_num = 4, 2
    policy_net = PolicyNet(env_dim=env_dim, action_num=action_num)
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=0.01)
    target_net = PolicyNet(env_dim=env_dim, action_num=action_num)

    agent = Agent(
        policy_net=policy_net,
        target_net=target_net,
        optimizer=optimizer,
        replay_buffer_size=1000,
    )

    # Test polyak update
    agent.polyak_update(beta=0.5)

    # Verify that target network parameters are updated
    for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
        assert not torch.equal(target_param.data, param.data)


def test_config():
    """Test the DqnConfig class."""
    config = DqnConfig()

    # Check default values
    assert config.env.env_name == "CartPole-v1"
    assert config.train.gamma == 0.999
    assert not config.train.use_target_network

    # Test custom config
    custom_config = DqnConfig(
        env=EnvConfig(env_name="MountainCar-v0", solved_threshold=90.0),
        train=TrainConfig(
            max_training_steps=1000,
            learning_rate=0.005,
            use_target_network=True,
            target_update_frequency=10,
        ),
    )

    assert custom_config.env.env_name == "MountainCar-v0"
    assert custom_config.env.solved_threshold == 90.0
    assert custom_config.train.max_training_steps == 1000
    assert custom_config.train.learning_rate == 0.005
    assert custom_config.train.use_target_network
    assert custom_config.train.target_update_frequency == 10


@pytest.mark.parametrize("use_target_network", [False, True])
def test_trainer_creation(use_target_network):
    """Test creating a trainer with both DQN and Double DQN variants."""
    config = DqnConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None),
        train=TrainConfig(
            max_training_steps=10,
            learning_rate=0.01,
            log_wandb=False,
            use_target_network=use_target_network,
        ),
    )

    trainer = DqnTrainer(config)

    assert isinstance(trainer.env, gym.Env)
    assert isinstance(trainer.agent, Agent)
    assert hasattr(trainer, "gamma")
    assert trainer.gamma == 0.999
    assert (trainer.agent._target_net is not None) == use_target_network


@pytest.mark.parametrize("use_target_network", [False, True])
def test_minimal_training(use_target_network):
    """Test minimal training run with a single episode for both DQN variants."""
    # Create minimal config with minimal steps
    config = DqnConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None),
        train=TrainConfig(
            max_training_steps=1,  # Just run a single step
            learning_starts=0,  # Start training immediately
            policy_update_frequency=1,  # Train every step
            learning_rate=0.01,
            log_wandb=False,
            use_target_network=use_target_network,
        ),
    )

    # Initialize trainer
    trainer = DqnTrainer(config)

    # Run training for one step
    trainer.train()

    # If we got here without errors, test passed
    assert True
