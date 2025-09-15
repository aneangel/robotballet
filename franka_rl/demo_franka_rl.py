#!/usr/bin/env python3
"""
Demo script for Franka RL Environment

Simple demonstration of the Franka robot learning to reach spheres.
This script shows how to use the environment and can be used to test
the setup before running full training.
"""

import time
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from franka_rl import FrankaRLEnvironment


def demo_random_actions():
    """Demonstrate the environment with random actions."""
    print("=" * 60)
    print("FRANKA RL ENVIRONMENT DEMO")
    print("=" * 60)
    print("Testing environment with random actions...")
    print("The robot will attempt to reach green spheres using random movements.")
    print("This demonstrates the environment setup before RL training.")
    print("=" * 60)
    
    # Create environment
    env = FrankaRLEnvironment(render_mode="human")
    
    try:
        for episode in range(3):
            print(f"\nEpisode {episode + 1}/3")
            
            # Reset environment
            obs, info = env.reset()
            print(f"Target sphere position: [{info['target_position'][0]:.3f}, "
                  f"{info['target_position'][1]:.3f}, {info['target_position'][2]:.3f}]")
            print(f"Initial distance: {info['initial_distance']:.3f}m")
            print(f"Goal tolerance: {env.goal_tolerance:.3f}m")
            
            episode_reward = 0
            best_distance = info['initial_distance']
            
            # Run episode
            for step in range(200):  # Limit steps for demo
                # Random action (the robot won't be very effective)
                action = env.action_space.sample() * 0.3  # Scale down for safety
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                # Track best distance achieved
                current_distance = info['distance_to_target']
                if current_distance < best_distance:
                    best_distance = current_distance
                
                # Render
                env.render()
                time.sleep(0.02)  # Small delay for visualization
                
                # Check if episode ended
                if terminated:
                    print(f"SUCCESS! Reached target in {step + 1} steps!")
                    print(f"Final distance: {current_distance:.4f}m")
                    break
                    
                if truncated:
                    print(f"Episode ended after {step + 1} steps (time limit)")
                    print(f"Final distance: {current_distance:.4f}m")
                    break
                    
                # Print progress occasionally
                if step % 50 == 0 and step > 0:
                    print(f"  Step {step}: distance = {current_distance:.4f}m, "
                          f"reward = {reward:.3f}")
            
            print(f"Episode summary:")
            print(f"  - Total reward: {episode_reward:.2f}")
            print(f"  - Best distance achieved: {best_distance:.4f}m")
            print(f"  - Success: {'Yes' if info.get('success', False) else 'No'}")
            
            # Short pause between episodes
            if episode < 2:
                print("\nPress Enter for next episode...")
                input()
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        env.close()
        print("\nDemo complete!")
        print("\nNext steps:")
        print("1. Run 'python train_franka_rl.py' to train with PPO")
        print("2. The robot will learn to reach spheres efficiently")
        print("3. Training progress will be logged to 'logs/franka_rl/'")


def demo_workspace_limits():
    """Demonstrate the workspace limits and target generation."""
    print("\n" + "=" * 60)
    print("WORKSPACE LIMITS DEMONSTRATION")
    print("=" * 60)
    
    env = FrankaRLEnvironment()
    
    print("Workspace limits for target sphere generation:")
    print(f"  X: {env.workspace_limits['x'][0]:.2f} to {env.workspace_limits['x'][1]:.2f} m")
    print(f"  Y: {env.workspace_limits['y'][0]:.2f} to {env.workspace_limits['y'][1]:.2f} m") 
    print(f"  Z: {env.workspace_limits['z'][0]:.2f} to {env.workspace_limits['z'][1]:.2f} m")
    
    print("\nGenerating 10 random target positions:")
    for i in range(10):
        obs, info = env.reset()
        target = info['target_position']
        distance_from_base = np.linalg.norm(target - np.array([0, 0, 0.333]))
        print(f"  Target {i+1}: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] "
              f"(distance from base: {distance_from_base:.3f}m)")
    
    env.close()
    print("\nAll targets are within the defined workspace limits.")
    print("This ensures the robot can always reach the target sphere.")


def demo_observation_space():
    """Demonstrate the observation space structure."""
    print("\n" + "=" * 60)
    print("OBSERVATION SPACE DEMONSTRATION")
    print("=" * 60)
    
    env = FrankaRLEnvironment()
    obs, info = env.reset()
    
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Action space shape: {env.action_space.shape}")
    
    print("\nObservation breakdown:")
    idx = 0
    
    # Joint positions (7)
    joint_pos = obs[idx:idx+7]
    print(f"  Joint positions (7): {joint_pos}")
    idx += 7
    
    # Joint velocities (7)
    joint_vel = obs[idx:idx+7]
    print(f"  Joint velocities (7): {joint_vel}")
    idx += 7
    
    # End-effector position (3)
    ee_pos = obs[idx:idx+3]
    print(f"  End-effector position (3): {ee_pos}")
    idx += 3
    
    # Target position (3)
    target_pos = obs[idx:idx+3]
    print(f"  Target position (3): {target_pos}")
    idx += 3
    
    # Distance to target (1)
    distance = obs[idx]
    print(f"  Distance to target (1): {distance:.4f}")
    idx += 1
    
    # Direction to target (3)
    direction = obs[idx:idx+3]
    print(f"  Direction to target (3): {direction}")
    
    print(f"\nAction space limits: [{env.action_space.low[0]:.2f}, {env.action_space.high[0]:.2f}]")
    print("Actions represent joint position increments (radians per step)")
    
    env.close()


if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Random actions demo (interactive)")
    print("2. Workspace limits demo")
    print("3. Observation space demo")
    print("4. All demos")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            demo_random_actions()
        elif choice == "2":
            demo_workspace_limits()
        elif choice == "3":
            demo_observation_space()
        elif choice == "4":
            demo_workspace_limits()
            demo_observation_space()
            demo_random_actions()
        else:
            print("Invalid choice. Running random actions demo...")
            demo_random_actions()
            
    except KeyboardInterrupt:
        print("\nDemo ended by user")
