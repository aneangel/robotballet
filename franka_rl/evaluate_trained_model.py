#!/usr/bin/env python3
"""
Evaluate and Visualize Trained Franka RL Model

This script loads a trained PPO model and runs it with visual rendering
to see how well the robot performs the sphere reaching task. You can
watch the robot use its learned policy to efficiently reach target spheres.

Usage:
    python evaluate_trained_model.py --model path/to/model.zip
    python evaluate_trained_model.py  # Uses best model automatically
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from franka_rl import FrankaRLEnvironment


def find_best_model(models_dir: str = "../models/franka_rl") -> Optional[str]:
    """
    Find the best trained model automatically.
    
    Args:
        models_dir: Directory containing saved models
        
    Returns:
        Path to best model or None if not found
    """
    models_path = Path(models_dir)
    
    # Look for best_model.zip first (saved by EvalCallback)
    best_model = models_path / "best_model.zip"
    if best_model.exists():
        return str(best_model)
    
    # Look for final_model.zip
    final_model = models_path / "final_model.zip"
    if final_model.exists():
        return str(final_model)
    
    # Look for any checkpoint files
    checkpoint_files = list(models_path.glob("franka_rl_checkpoint_*.zip"))
    if checkpoint_files:
        # Return the most recent checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        return str(latest_checkpoint)
    
    return None


def evaluate_model_visually(
    model_path: str,
    n_episodes: int = 5,
    max_steps_per_episode: int = 300,
    goal_tolerance: float = 0.025,
    pause_between_episodes: bool = True,
    show_statistics: bool = True,
):
    """
    Evaluate the trained model with visual rendering.
    
    Args:
        model_path: Path to the trained model
        n_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        goal_tolerance: Distance tolerance for success
        pause_between_episodes: Whether to pause between episodes
        show_statistics: Whether to print detailed statistics
    """
    print("=" * 80)
    print("FRANKA RL MODEL EVALUATION - VISUAL DEMONSTRATION")
    print("=" * 80)
    print(f"Loading model: {model_path}")
    print(f"Episodes to run: {n_episodes}")
    print(f"Goal tolerance: {goal_tolerance:.3f}m")
    print("=" * 80)
    
    # Load the trained model
    try:
        model = PPO.load(model_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Create environment with visual rendering
    env = FrankaRLEnvironment(
        max_episode_steps=max_steps_per_episode,
        goal_tolerance=goal_tolerance,
        render_mode="human"  # Enable visual rendering
    )
    
    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    final_distances = []
    trajectory_efficiency = []
    
    try:
        for episode in range(n_episodes):
            print(f"\n{'='*20} EPISODE {episode + 1}/{n_episodes} {'='*20}")
            
            # Reset environment
            obs, info = env.reset()
            target_pos = info['target_position']
            initial_distance = info['initial_distance']
            
            print(f"Target sphere: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            print(f"Initial distance: {initial_distance:.3f}m")
            print("Watching robot learn to reach the target...")
            
            episode_reward = 0
            step_count = 0
            path_length = 0
            prev_ee_pos = info['ee_position']
            best_distance = initial_distance
            
            # Run episode
            while step_count < max_steps_per_episode:
                # Get action from trained policy (deterministic for evaluation)
                action, _ = model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                # Track robot movement
                current_ee_pos = info['ee_position']
                step_distance = np.linalg.norm(current_ee_pos - prev_ee_pos)
                path_length += step_distance
                prev_ee_pos = current_ee_pos
                
                # Track best distance achieved
                current_distance = info['distance_to_target']
                if current_distance < best_distance:
                    best_distance = current_distance
                
                # Render the environment
                env.render()
                time.sleep(0.03)  # Slight delay for smooth visualization
                
                # Print progress occasionally
                if step_count % 50 == 0:
                    print(f"  Step {step_count}: distance = {current_distance:.4f}m, "
                          f"reward = {reward:.3f}")
                
                # Check if episode ended
                if terminated:
                    success_count += 1
                    print(f"🎯 SUCCESS! Robot reached target in {step_count} steps!")
                    print(f"   Final distance: {current_distance:.4f}m")
                    break
                    
                if truncated:
                    print(f"⏱️  Episode ended after {step_count} steps (time limit)")
                    print(f"   Final distance: {current_distance:.4f}m")
                    break
            
            # Calculate trajectory efficiency
            straight_line_distance = initial_distance
            efficiency = straight_line_distance / max(path_length, 1e-6)
            
            # Store statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            final_distances.append(current_distance)
            trajectory_efficiency.append(efficiency)
            
            # Print episode summary
            if show_statistics:
                print(f"\nEpisode {episode + 1} Summary:")
                print(f"  ✓ Success: {'Yes' if terminated else 'No'}")
                print(f"  ✓ Total reward: {episode_reward:.2f}")
                print(f"  ✓ Steps taken: {step_count}")
                print(f"  ✓ Best distance: {best_distance:.4f}m")
                print(f"  ✓ Path length: {path_length:.3f}m")
                print(f"  ✓ Trajectory efficiency: {efficiency:.3f}")
            
            # Pause between episodes if requested
            if pause_between_episodes and episode < n_episodes - 1:
                print(f"\nPress Enter to continue to episode {episode + 2}...")
                input()
    
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    
    finally:
        env.close()
    
    # Print final statistics
    if episode_rewards:
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Episodes completed: {len(episode_rewards)}")
        print(f"Success rate: {success_count}/{len(episode_rewards)} ({success_count/len(episode_rewards):.1%})")
        print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"Average final distance: {np.mean(final_distances):.4f} ± {np.std(final_distances):.4f}m")
        print(f"Average trajectory efficiency: {np.mean(trajectory_efficiency):.3f} ± {np.std(trajectory_efficiency):.3f}")
        
        if success_count > 0:
            successful_episodes = [i for i, terminated in enumerate([r > 0 for r in episode_rewards]) if terminated]
            if successful_episodes:
                successful_lengths = [episode_lengths[i] for i in successful_episodes]
                successful_rewards = [episode_rewards[i] for i in successful_episodes]
                print(f"\nSuccessful episodes only:")
                print(f"  Average steps: {np.mean(successful_lengths):.1f}")
                print(f"  Average reward: {np.mean(successful_rewards):.2f}")
        
        print("=" * 80)


def run_continuous_evaluation(model_path: str, goal_tolerance: float = 0.025):
    """
    Run continuous evaluation where the robot keeps trying different targets.
    
    Args:
        model_path: Path to the trained model
        goal_tolerance: Distance tolerance for success
    """
    print("=" * 80)
    print("CONTINUOUS EVALUATION MODE")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = FrankaRLEnvironment(
        max_episode_steps=300,
        goal_tolerance=goal_tolerance,
        render_mode="human"
    )
    
    episode_count = 0
    success_count = 0
    
    try:
        while True:
            episode_count += 1
            print(f"\nContinuous Episode {episode_count}")
            
            obs, info = env.reset()
            target_pos = info['target_position']
            print(f"New target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            
            step_count = 0
            while step_count < 300:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                
                env.render()
                time.sleep(0.02)
                
                if terminated:
                    success_count += 1
                    print(f"✓ Success! ({success_count}/{episode_count} = {success_count/episode_count:.1%})")
                    time.sleep(1.0)  # Brief pause to see success
                    break
                    
                if truncated:
                    print(f"✗ Failed in {step_count} steps")
                    break
    
    except KeyboardInterrupt:
        print(f"\nStopped after {episode_count} episodes")
        print(f"Final success rate: {success_count}/{episode_count} ({success_count/episode_count:.1%})")
    
    finally:
        env.close()


def main():
    """Main entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained Franka RL model")
    parser.add_argument("--model", type=str, help="Path to trained model (.zip file)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous mode")
    parser.add_argument("--no-pause", action="store_true", help="Don't pause between episodes")
    parser.add_argument("--tolerance", type=float, default=0.025, help="Goal tolerance in meters")
    
    args = parser.parse_args()
    
    # Find model path
    if args.model:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return
    else:
        print("Searching for best trained model...")
        model_path = find_best_model()
        if model_path is None:
            print("Error: No trained model found!")
            print("Please train a model first using: python train_franka_rl.py")
            print("Or specify a model path with: --model path/to/model.zip")
            return
        print(f"Found model: {model_path}")
    
    # Run evaluation
    if args.continuous:
        run_continuous_evaluation(model_path, args.tolerance)
    else:
        evaluate_model_visually(
            model_path=model_path,
            n_episodes=args.episodes,
            goal_tolerance=args.tolerance,
            pause_between_episodes=not args.no_pause,
        )


if __name__ == "__main__":
    main()
