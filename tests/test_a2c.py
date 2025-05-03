"""Tests for the A2C algorithm."""

import gymnasium as gym
import numpy as np
import torch

from toyrl.a2c import (
    A2CConfig,
    A2CTrainer,
    ActorCriticNet,
    Agent,
    EnvConfig,
    Experience,
    ReplayBuffer,
    TrainConfig,
)


def test_actor_critic_net():
    """Test the ActorCriticNet class."""
    env_dim, action_num = 4, 2
    net = ActorCriticNet(env_dim=env_dim, action_num=action_num)

    # Test forward pass
    x = torch.randn(env_dim, dtype=torch.float32)
    policy_logits, value = net(x)

    assert policy_logits.shape == torch.Size([action_num])
    assert value.shape == torch.Size([1])
    assert isinstance(policy_logits, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    assert policy_logits.dtype == torch.float32
    assert value.dtype == torch.float32


def test_replay_buffer():
    """Test the ReplayBuffer class."""
    buffer = ReplayBuffer()

    # Test empty buffer
    assert len(buffer) == 0
    assert buffer.total_reward() == 0

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
    assert buffer.total_reward() == 1.0

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

    assert len(buffer) == 2
    assert buffer.total_reward() == 1.5

    # Test sampling
    samples = buffer.sample()
    assert len(samples) == 2
    assert samples[0].action == 0
    assert samples[1].reward == 0.5

    # Test reset
    buffer.reset()
    assert len(buffer) == 0
    assert buffer.total_reward() == 0


def test_agent_creation():
    """Test creating an agent."""
    env_dim, action_num = 4, 2
    net = ActorCriticNet(env_dim=env_dim, action_num=action_num)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    agent = Agent(net=net, optimizer=optimizer)

    # Check agent properties
    assert hasattr(agent, "_net")
    assert hasattr(agent, "_optimizer")
    assert hasattr(agent, "_replay_buffer")


def test_agent_act():
    """Test the agent's act method."""
    env_dim, action_num = 4, 2
    net = ActorCriticNet(env_dim=env_dim, action_num=action_num)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    agent = Agent(net=net, optimizer=optimizer)

    # Test act method in training mode
    observation = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    action = agent.act(observation)

    assert isinstance(action, int)
    assert action in range(action_num)  # For CartPole, should be 0 or 1

    # Test act method in eval mode
    action = agent.act(observation, eval=True)
    assert isinstance(action, int)
    assert action in range(action_num)


def test_config():
    """Test the Config class."""
    config = A2CConfig()

    # Check default values
    assert config.env.env_name == "CartPole-v1"
    assert config.train.gamma == 0.999
    assert config.train.lambda_ == 0.98
    assert config.train.value_loss_coef == 0.5
    assert config.train.policy_loss_coef == 0.5
    assert config.train.entropy_coef == 0.01
    assert config.train.eval_episodes == 10
    assert config.train.eval_interval == 100

    # Test custom config
    custom_config = A2CConfig(
        env=EnvConfig(env_name="MountainCar-v0", solved_threshold=90.0),
        train=TrainConfig(
            num_episodes=1000,
            learning_rate=0.005,
            gamma=0.99,
            lambda_=0.95,
            value_loss_coef=0.7,
            policy_loss_coef=0.8,
            entropy_coef=0.02,
            eval_episodes=5,
            eval_interval=50,
        ),
    )

    assert custom_config.env.env_name == "MountainCar-v0"
    assert custom_config.env.solved_threshold == 90.0
    assert custom_config.train.num_episodes == 1000
    assert custom_config.train.learning_rate == 0.005
    assert custom_config.train.gamma == 0.99
    assert custom_config.train.lambda_ == 0.95
    assert custom_config.train.value_loss_coef == 0.7
    assert custom_config.train.policy_loss_coef == 0.8
    assert custom_config.train.entropy_coef == 0.02
    assert custom_config.train.eval_episodes == 5
    assert custom_config.train.eval_interval == 50


def test_trainer_creation():
    """Test creating a trainer."""
    config = A2CConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None),
        train=TrainConfig(num_episodes=10, learning_rate=0.01, log_wandb=False),
    )

    trainer = A2CTrainer(config)

    assert isinstance(trainer.env, gym.Env)
    assert isinstance(trainer.agent, Agent)
    assert hasattr(trainer, "num_episodes")
    assert trainer.num_episodes == 10
    assert trainer.gamma == 0.999
    assert trainer.lambda_ == 0.98
    assert trainer.value_loss_coef == 0.5
    assert trainer.policy_loss_coef == 0.5
    assert trainer.entropy_coef == 0.01


def test_agent_net_update():
    """Test the agent's net_update method."""
    env_dim, action_num = 4, 2
    net = ActorCriticNet(env_dim=env_dim, action_num=action_num)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    agent = Agent(net=net, optimizer=optimizer)

    # Add some experiences
    exp1 = Experience(
        observation=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action=0,
        reward=1.0,
        next_observation=np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        terminated=False,
        truncated=False,
    )
    agent.add_experience(exp1)

    exp2 = Experience(
        observation=np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        action=1,
        reward=0.5,
        next_observation=np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32),
        terminated=True,
        truncated=False,
    )
    agent.add_experience(exp2)

    # Test net_update
    loss, entropy, adv_mean = agent.net_update(
        gamma=0.99, lambda_=0.95, value_loss_coef=0.5, policy_loss_coef=0.5, entropy_coef=0.01
    )

    assert isinstance(loss, float)
    assert isinstance(entropy, float)
    assert isinstance(adv_mean, float)

    # Check that buffer is still intact
    assert len(agent._replay_buffer) == 2


def test_minimal_training():
    """Test minimal training run with a single episode."""
    # Create minimal config with just one episode
    config = A2CConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None),
        train=TrainConfig(num_episodes=1, learning_rate=0.01, log_wandb=False),
    )

    # Initialize trainer
    trainer = A2CTrainer(config)

    # Run training for one episode
    trainer.train()

    # If we got here without errors, test passed
    assert True


def test_trainer_evaluation():
    """Test the evaluation functionality of the trainer."""
    config = A2CConfig(
        env=EnvConfig(env_name="CartPole-v1", render_mode=None),
        train=TrainConfig(num_episodes=1, learning_rate=0.01, log_wandb=False, eval_episodes=2, eval_interval=1),
    )

    trainer = A2CTrainer(config)

    # Test evaluation method
    eval_reward = trainer.evaluate(num_episodes=2)

    assert isinstance(eval_reward, float)
    # For a randomly initialized agent, we expect some reward but usually less than 50
    # per episode for CartPole
    assert 0 <= eval_reward <= 100
