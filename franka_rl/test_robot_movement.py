#!/usr/bin/env python3
"""
Test Robot Movement

Quick test to verify that the robot actually moves when we send actions.
This will help confirm the actuator fix is working before running full training.
"""

import time
import numpy as np
import sys
import os

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from franka_rl import FrankaRLEnvironment


def test_robot_movement():
    """Test that the robot actually moves when we send actions."""
    print("=" * 60)
    print("ROBOT MOVEMENT TEST")
    print("=" * 60)
    print("Testing that the robot responds to actions...")
    
    # Create environment
    env = FrankaRLEnvironment(render_mode="human")
    
    try:
        # Reset environment
        obs, info = env.reset()
        print(f"Model has {env.model.nu} actuators")
        print(f"Control vector size: {env.data.ctrl.shape}")
        
        # Get initial state
        initial_ee_pos = env.data.site_xpos[env.ee_site_id].copy()
        initial_joint_pos = np.array([env.data.qpos[joint_id] for joint_id in env.robot_joints])
        
        print(f"Initial EE position: {initial_ee_pos}")
        print(f"Initial joint positions: {initial_joint_pos}")
        
        print("\nApplying test actions...")
        
        # Test 1: Apply a specific action and see if robot moves
        test_action = np.array([0.1, -0.1, 0.05, 0.1, 0, 0, 0])  # Move first few joints
        print(f"Test action: {test_action}")
        
        # Apply action for several steps
        for step in range(50):
            obs, reward, terminated, truncated, info = env.step(test_action)
            env.render()
            time.sleep(0.05)  # Slower for observation
            
            if step % 10 == 0:
                current_ee_pos = env.data.site_xpos[env.ee_site_id].copy()
                current_joint_pos = np.array([env.data.qpos[joint_id] for joint_id in env.robot_joints])
                
                ee_movement = np.linalg.norm(current_ee_pos - initial_ee_pos)
                joint_movement = np.linalg.norm(current_joint_pos - initial_joint_pos)
                
                print(f"Step {step}: EE moved {ee_movement:.4f}m, joints moved {joint_movement:.4f}rad")
                print(f"  Current EE: {current_ee_pos}")
                print(f"  Distance to target: {info['distance_to_target']:.4f}m")
        
        # Final check
        final_ee_pos = env.data.site_xpos[env.ee_site_id].copy()
        final_joint_pos = np.array([env.data.qpos[joint_id] for joint_id in env.robot_joints])
        
        total_ee_movement = np.linalg.norm(final_ee_pos - initial_ee_pos)
        total_joint_movement = np.linalg.norm(final_joint_pos - initial_joint_pos)
        
        print("\n" + "=" * 60)
        print("MOVEMENT TEST RESULTS")
        print("=" * 60)
        print(f"Total EE movement: {total_ee_movement:.4f} meters")
        print(f"Total joint movement: {total_joint_movement:.4f} radians")
        
        if total_ee_movement > 0.01:  # Moved at least 1cm
            print("✅ SUCCESS: Robot is moving in response to actions!")
        else:
            print("❌ FAILURE: Robot is not moving. Check actuator setup.")
            
        if total_joint_movement > 0.1:  # Moved at least 0.1 radians
            print("✅ SUCCESS: Joints are responding to control signals!")
        else:
            print("❌ FAILURE: Joints are not moving. Check actuator mapping.")
            
        print("=" * 60)
        
        # Test 2: Random actions
        print("\nTesting with random actions (10 steps)...")
        for step in range(10):
            random_action = env.action_space.sample() * 0.5  # Scale down for safety
            obs, reward, terminated, truncated, info = env.step(random_action)
            env.render()
            time.sleep(0.1)
            
            current_distance = info['distance_to_target']
            print(f"Random step {step}: distance = {current_distance:.4f}m, reward = {reward:.3f}")
            
            if terminated:
                print("🎯 Reached target with random actions!")
                break
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        env.close()
        print("\nMovement test complete!")


if __name__ == "__main__":
    test_robot_movement()
