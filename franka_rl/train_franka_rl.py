#!/usr/bin/env python3
"""
Train Franka Robot with PPO for Sphere Reaching

This script trains a Franka Panda robot to reach randomly positioned spheres
using Proximal Policy Optimization (PPO). The training uses the kinematic
principles from franka_kinematic_visualizer.py but learns the control policy
through reinforcement learning.

Features:
- PPO training with optimized hyperparameters for manipulation tasks
- Curriculum learning for progressive difficulty
- Success rate monitoring and logging
- Model checkpointing and evaluation
- Tensorboard integration for training visualization
"""

import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import matplotlib.pyplot as plt

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from franka_rl import FrankaRLEnvironment


@dataclass
class TrainingConfig:
    """Configuration for PPO training."""
    
    # Environment settings
    n_envs: int = 4  # Number of parallel environments
    max_episode_steps: int = 300
    goal_tolerance: float = 0.025  # 2.5cm tolerance
    
    # PPO hyperparameters (optimized for manipulation tasks)
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 2048  # Steps per rollout
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01  # Encourage exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Network architecture
    policy_kwargs: Dict = None
    
    # Training settings
    save_freq: int = 10_000  # Save model every N steps
    eval_freq: int = 5_000   # Evaluate every N steps
    eval_episodes: int = 10  # Episodes per evaluation
    
    # Paths
    log_dir: str = "logs/franka_rl"
    model_dir: str = "models/franka_rl"
    
    def __post_init__(self):
        """Set default policy kwargs."""
        if self.policy_kwargs is None:
            self.policy_kwargs = {
                "net_arch": [256, 256],  # Two hidden layers with 256 units each
                "activation_fn": torch.nn.ReLU,
                "ortho_init": False,
            }


class SuccessRateCallback(BaseCallback):
    """
    Callback for monitoring success rate during training.
    """
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.success_count = 0
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        """Called at each step."""
        # Check if any environment finished an episode
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                self.episode_count += 1
                # Check if episode was successful
                info = self.locals.get("infos", [{}])[i]
                if info.get("success", False):
                    self.success_count += 1
                    
        # Log success rate periodically
        if self.episode_count > 0 and self.episode_count % self.check_freq == 0:
            success_rate = self.success_count / self.episode_count
            self.logger.record("success/success_rate", success_rate)
            self.logger.record("success/total_episodes", self.episode_count)
            self.logger.record("success/successful_episodes", self.success_count)
            
            if self.verbose > 0:
                print(f"Success rate: {success_rate:.3f} ({self.success_count}/{self.episode_count})")
                
        return True


class ProgressCallback(BaseCallback):
    """
    Callback for monitoring training progress and adaptive learning.
    """
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.distances_to_target = []
        
    def _on_step(self) -> bool:
        """Called at each step."""
        # Collect episode statistics
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [{}])[i]
                
                self.episode_rewards.append(info.get("episode_return", 0))
                self.episode_lengths.append(info.get("episode_step", 0))
                self.distances_to_target.append(info.get("distance_to_target", float('inf')))
                
        # Log statistics periodically
        if len(self.episode_rewards) > 0 and len(self.episode_rewards) % self.check_freq == 0:
            avg_reward = np.mean(self.episode_rewards[-self.check_freq:])
            avg_length = np.mean(self.episode_lengths[-self.check_freq:])
            avg_distance = np.mean(self.distances_to_target[-self.check_freq:])
            
            self.logger.record("progress/avg_episode_reward", avg_reward)
            self.logger.record("progress/avg_episode_length", avg_length)
            self.logger.record("progress/avg_final_distance", avg_distance)
            
            if self.verbose > 0:
                print(f"Progress - Avg reward: {avg_reward:.2f}, "
                      f"Avg length: {avg_length:.1f}, "
                      f"Avg final distance: {avg_distance:.4f}m")
                
        return True


def make_env(config: TrainingConfig, rank: int = 0, seed: int = 0) -> gym.Env:
    """
    Create a single environment instance.
    
    Args:
        config: Training configuration
        rank: Environment rank (for parallel envs)
        seed: Random seed
        
    Returns:
        Monitored environment
    """
    def _init():
        env = FrankaRLEnvironment(
            max_episode_steps=config.max_episode_steps,
            goal_tolerance=config.goal_tolerance,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)
    
    return _init


def create_training_environment(config: TrainingConfig) -> gym.Env:
    """
    Create vectorized training environment.
    
    Args:
        config: Training configuration
        
    Returns:
        Vectorized environment
    """
    if config.n_envs == 1:
        return DummyVecEnv([make_env(config, 0)])
    else:
        return SubprocVecEnv([make_env(config, i) for i in range(config.n_envs)])


def evaluate_policy(model: PPO, env: gym.Env, n_episodes: int = 10) -> Dict:
    """
    Evaluate trained policy.
    
    Args:
        model: Trained PPO model
        env: Environment for evaluation
        n_episodes: Number of episodes to evaluate
        
    Returns:
        Evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    final_distances = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                final_distances.append(info.get("distance_to_target", float('inf')))
                
                if info.get("success", False):
                    success_count += 1
                    
                break
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": success_count / n_episodes,
        "mean_final_distance": np.mean(final_distances),
        "episode_rewards": episode_rewards,
    }


def plot_training_results(log_dir: Path):
    """
    Plot training results from tensorboard logs.
    
    Args:
        log_dir: Directory containing tensorboard logs
    """
    try:
        import tensorboard as tb
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Find the latest tensorboard log file
        log_files = list(log_dir.glob("**/events.out.tfevents.*"))
        if not log_files:
            print("No tensorboard log files found")
            return
            
        latest_log = max(log_files, key=os.path.getctime)
        
        # Load tensorboard data
        ea = EventAccumulator(str(latest_log))
        ea.Reload()
        
        # Extract scalar data
        scalar_tags = ea.Tags()['scalars']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Franka RL Training Progress')
        
        # Plot rewards if available
        if 'rollout/ep_rew_mean' in scalar_tags:
            reward_data = ea.Scalars('rollout/ep_rew_mean')
            steps = [x.step for x in reward_data]
            rewards = [x.value for x in reward_data]
            axes[0, 0].plot(steps, rewards)
            axes[0, 0].set_title('Episode Reward')
            axes[0, 0].set_xlabel('Training Steps')
            axes[0, 0].set_ylabel('Mean Reward')
            
        # Plot success rate if available
        if 'success/success_rate' in scalar_tags:
            success_data = ea.Scalars('success/success_rate')
            steps = [x.step for x in success_data]
            success_rates = [x.value for x in success_data]
            axes[0, 1].plot(steps, success_rates)
            axes[0, 1].set_title('Success Rate')
            axes[0, 1].set_xlabel('Training Steps')
            axes[0, 1].set_ylabel('Success Rate')
            
        # Plot episode length if available
        if 'rollout/ep_len_mean' in scalar_tags:
            length_data = ea.Scalars('rollout/ep_len_mean')
            steps = [x.step for x in length_data]
            lengths = [x.value for x in length_data]
            axes[1, 0].plot(steps, lengths)
            axes[1, 0].set_title('Episode Length')
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('Mean Length')
            
        # Plot learning rate if available
        if 'train/learning_rate' in scalar_tags:
            lr_data = ea.Scalars('train/learning_rate')
            steps = [x.step for x in lr_data]
            learning_rates = [x.value for x in lr_data]
            axes[1, 1].plot(steps, learning_rates)
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Training Steps')
            axes[1, 1].set_ylabel('Learning Rate')
            
        plt.tight_layout()
        plt.savefig(log_dir / "training_progress.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Training plots saved to {log_dir / 'training_progress.png'}")
        
    except ImportError:
        print("Tensorboard not available for plotting")
    except Exception as e:
        print(f"Could not plot results: {e}")


def train_franka_rl(config: TrainingConfig):
    """
    Train Franka robot to reach spheres using PPO.
    
    Args:
        config: Training configuration
    """
    print("=" * 60)
    print("FRANKA RL TRAINING - SPHERE REACHING WITH PPO")
    print("=" * 60)
    print(f"Training configuration:")
    print(f"  - Total timesteps: {config.total_timesteps:,}")
    print(f"  - Parallel environments: {config.n_envs}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Goal tolerance: {config.goal_tolerance}m")
    print(f"  - Max episode steps: {config.max_episode_steps}")
    print("=" * 60)
    
    # Create directories
    log_dir = Path(config.log_dir)
    model_dir = Path(config.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training environment
    print("Creating training environment...")
    train_env = create_training_environment(config)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = FrankaRLEnvironment(
        max_episode_steps=config.max_episode_steps,
        goal_tolerance=config.goal_tolerance,
    )
    eval_env = Monitor(eval_env)
    
    # Configure logging
    logger = configure(str(log_dir), ["stdout", "tensorboard"])
    
    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=config.policy_kwargs,
        verbose=1,
        tensorboard_log=str(log_dir),
    )
    
    model.set_logger(logger)
    
    # Setup callbacks
    callbacks = [
        SuccessRateCallback(check_freq=1000),
        ProgressCallback(check_freq=1000),
        CheckpointCallback(
            save_freq=config.save_freq,
            save_path=str(model_dir),
            name_prefix="franka_rl_checkpoint"
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(log_dir),
            eval_freq=config.eval_freq,
            n_eval_episodes=config.eval_episodes,
            deterministic=True,
            render=False,
        )
    ]
    
    # Train the model
    print("\nStarting training...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.1f} seconds")
        
        # Save final model
        final_model_path = model_dir / "final_model.zip"
        model.save(str(final_model_path))
        print(f"Final model saved to {final_model_path}")
        
        # Final evaluation
        print("\nPerforming final evaluation...")
        eval_results = evaluate_policy(model, eval_env, n_episodes=20)
        
        print("\nFinal Evaluation Results:")
        print(f"  - Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"  - Success rate: {eval_results['success_rate']:.1%}")
        print(f"  - Mean episode length: {eval_results['mean_length']:.1f}")
        print(f"  - Mean final distance: {eval_results['mean_final_distance']:.4f}m")
        
        # Plot results
        plot_training_results(log_dir)
        
        # Save training summary
        training_summary = {
            "config": config.__dict__,
            "training_time": training_time,
            "final_evaluation": eval_results,
        }
        
        import json
        with open(log_dir / "training_summary.json", "w") as f:
            json.dump(training_summary, f, indent=2, default=str)
            
        print(f"\nTraining summary saved to {log_dir / 'training_summary.json'}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        model.save(str(model_dir / "interrupted_model.zip"))
        print(f"Model saved to {model_dir / 'interrupted_model.zip'}")
        
    finally:
        # Clean up environments
        train_env.close()
        eval_env.close()
        
        print("\nTraining complete!")


def main():
    """Main entry point."""
    # Create default configuration
    config = TrainingConfig()
    
    # Train the model
    train_franka_rl(config)


if __name__ == "__main__":
    main()
