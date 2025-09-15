#!/usr/bin/env python3
"""
Franka RL Environment for Sphere Reaching

A reinforcement learning environment where a Franka Panda robot learns to reach
randomly positioned spheres using PPO. Built on the kinematic methods from
franka_kinematic_visualizer.py while maintaining the original structure.

The environment generates random target spheres within the robot's reachable
workspace and trains the robot to efficiently reach them using inverse kinematics
principles learned through reinforcement learning.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial.transform import Rotation
import time


class FrankaRLEnvironment(gym.Env):
    """
    Reinforcement Learning environment for Franka robot sphere reaching.
    
    The robot learns to reach randomly positioned target spheres using PPO.
    Uses kinematic principles from franka_kinematic_visualizer.py but learns
    the control policy through RL instead of analytical inverse kinematics.
    
    Observation Space:
    - Joint angles (7)
    - Joint velocities (7) 
    - End-effector position (3)
    - Target sphere position (3)
    - Distance to target (1)
    - Direction vector to target (3)
    Total: 24 dimensions
    
    Action Space:
    - Joint position increments (7) - small deltas applied each step
    """
    
    def __init__(
        self,
        menagerie_path: str = None,
        max_episode_steps: int = 300,
        control_timestep: float = 0.02,
        render_mode: Optional[str] = None,
        goal_tolerance: float = 0.025,  # 2.5cm tolerance for reaching sphere
        workspace_limits: Dict[str, Tuple[float, float]] = None,
    ):
        """
        Initialize the Franka RL environment.
        
        Args:
            menagerie_path: Path to mujoco_menagerie repository
            max_episode_steps: Maximum steps per episode
            control_timestep: Control frequency
            render_mode: Visualization mode
            goal_tolerance: Distance tolerance for successful reach
            workspace_limits: Workspace bounds for target generation
        """
        super().__init__()
        
        # Set default menagerie path
        if menagerie_path is None:
            self.menagerie_path = os.path.expanduser("~/code/mujoco_menagerie")
        else:
            self.menagerie_path = menagerie_path
            
        self.max_episode_steps = max_episode_steps
        self.control_timestep = control_timestep
        self.render_mode = render_mode
        self.goal_tolerance = goal_tolerance
        
        # Default workspace limits (reachable area around robot base)
        if workspace_limits is None:
            self.workspace_limits = {
                'x': (0.2, 0.7),   # Forward reach
                'y': (-0.4, 0.4),  # Left-right reach
                'z': (0.1, 0.6)    # Height above table
            }
        else:
            self.workspace_limits = workspace_limits
            
        # Episode tracking
        self.current_step = 0
        self.episode_return = 0.0
        self.total_episodes = 0
        
        # Target tracking
        self.current_target = None
        self.target_body_id = None
        self.initial_distance = None
        self.prev_distance = None
        
        # Robot tracking
        self.robot_joints = []
        self.robot_actuators = []
        self.ee_site_id = None
        
        # Simulation objects
        self.model = None
        self.data = None
        self.viewer = None
        
        self._setup_simulation()
        self._setup_robot_info()
        self._setup_spaces()
        
    def _setup_simulation(self):
        """
        Build the MuJoCo simulation using methods from franka_kinematic_visualizer.
        
        Creates the robot model with target sphere, maintaining the original
        structure from the kinematic visualizer.
        """
        franka_model_path = os.path.join(self.menagerie_path, "franka_emika_panda", "panda.xml")
        
        with open(franka_model_path, 'r') as f:
            franka_xml = f.read()
        
        # Update mesh paths to absolute paths (from kinematic visualizer)
        xml_content = franka_xml.replace(
            'meshdir="assets"',
            f'meshdir="{self.menagerie_path}/franka_emika_panda/assets"'
        )
        
        # Add tracking site to robot hand (from kinematic visualizer)
        xml_content = xml_content.replace(
            '<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">',
            '''<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                  <site name="end_effector" pos="0 0 0.103" size="0.02" rgba="1 0 0 1"/>'''
        )
        
        # Add scene elements (modified from kinematic visualizer for sphere target)
        xml_content = xml_content.replace('</worldbody>', f'''
            <!-- Scene lighting -->
            <light name="main" pos="0 0 3" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" directional="false"/>
            <light name="fill1" pos="1 1 2" diffuse="0.4 0.4 0.4" directional="false"/>
            <light name="fill2" pos="-1 -1 2" diffuse="0.4 0.4 0.4" directional="false"/>
            
            <!-- Floor -->
            <geom name="floor" type="plane" pos="0 0 0" size="3 3 0.1" rgba="0.95 0.95 0.95 1"/>
            
            <!-- Target sphere (starts at default position, will be moved during reset) -->
            <body name="target_sphere" pos="0.5 0.0 0.3">
              <geom name="sphere_target" type="sphere" size="0.02" rgba="0 1 0 0.8"/>
              <site name="target_site" pos="0 0 0" size="0.025" rgba="0 1 0 1"/>
            </body>
            
            <!-- End-effector trail visualization (optional) -->
            <body name="ee_trail" pos="0 0 -1">
              <geom name="trail_sphere" type="sphere" size="0.005" rgba="1 0 0 0.3"/>
            </body>
          </worldbody>''')
        
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.data = mujoco.MjData(self.model)
        
        # Set neutral pose (from kinematic visualizer)
        self._set_neutral_pose()
        
        print("Franka RL Environment initialized")
        print(f"Robot model: {self.menagerie_path}")
        print(f"Workspace limits: {self.workspace_limits}")
        
    def _set_neutral_pose(self):
        """Set robot to starting pose (from kinematic visualizer)."""
        neutral_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        
        for i in range(7):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            if joint_id >= 0:
                self.data.qpos[joint_id] = neutral_qpos[i]
        
        mujoco.mj_forward(self.model, self.data)
        
    def _setup_robot_info(self):
        """Find robot joint and actuator IDs."""
        self.robot_joints = []
        self.robot_actuators = []
        
        # Find joint IDs
        for i in range(7):
            joint_name = f"joint{i+1}"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                self.robot_joints.append(joint_id)
            else:
                print(f"WARNING: Could not find joint {joint_name}")
                
        # Find actuator IDs
        for i in range(7):
            actuator_name = f"actuator_{i+1}"
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id >= 0:
                self.robot_actuators.append(actuator_id)
            else:
                print(f"WARNING: Could not find actuator {actuator_name}")
                
        # Find end-effector site
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        if self.ee_site_id < 0:
            print("WARNING: Could not find end_effector site")
            
        # Find target sphere body
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_sphere")
        if self.target_body_id < 0:
            print("WARNING: Could not find target_sphere body")
            
        print(f"Found {len(self.robot_joints)} joints and {len(self.robot_actuators)} actuators")
        
    def _setup_spaces(self):
        """Define observation and action spaces."""
        # Observation space: joints(7) + velocities(7) + ee_pos(3) + target_pos(3) + distance(1) + direction(3) = 24
        obs_dim = 24
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: Joint position increments (7 DOF)
        # Use moderate increments for smooth learning
        max_joint_delta = 0.1  # rad per step
        self.action_space = spaces.Box(
            low=-max_joint_delta, high=max_joint_delta, shape=(7,), dtype=np.float32
        )
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment with new random target sphere."""
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_return = 0.0
        self.total_episodes += 1
        
        # Reset robot to neutral pose with small perturbation
        neutral_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        
        for i, joint_id in enumerate(self.robot_joints):
            if i < len(neutral_qpos):
                # Add small random perturbation to avoid identical starts
                perturbation = np.random.normal(0, 0.03)
                self.data.qpos[joint_id] = neutral_qpos[i] + perturbation
        
        # Reset velocities
        self.data.qvel[:] = 0.0
        
        # Generate new random target sphere position
        self._generate_random_target()
        
        # Step physics to settle
        mujoco.mj_forward(self.model, self.data)
        
        # Calculate initial distance
        ee_pos = self.data.site_xpos[self.ee_site_id]
        self.initial_distance = np.linalg.norm(ee_pos - self.current_target)
        self.prev_distance = self.initial_distance
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
        
    def _generate_random_target(self):
        """
        Generate a random target sphere position within reachable workspace.
        
        Uses workspace limits to ensure targets are always reachable,
        following the constraint from the user requirements.
        """
        # Generate random position within workspace limits
        x = np.random.uniform(self.workspace_limits['x'][0], self.workspace_limits['x'][1])
        y = np.random.uniform(self.workspace_limits['y'][0], self.workspace_limits['y'][1])
        z = np.random.uniform(self.workspace_limits['z'][0], self.workspace_limits['z'][1])
        
        self.current_target = np.array([x, y, z])
        
        # Move the target sphere body to new position
        if self.target_body_id >= 0:
            self.model.body_pos[self.target_body_id] = self.current_target
            
        print(f"New target sphere at: [{x:.3f}, {y:.3f}, {z:.3f}]")
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Apply action
        self._apply_action(action)
        
        # Step simulation
        n_sim_steps = int(self.control_timestep / self.model.opt.timestep)
        for _ in range(n_sim_steps):
            mujoco.mj_step(self.model, self.data)
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._calculate_reward(action)
        self.episode_return += reward
        
        # Check termination
        self.current_step += 1
        terminated = self._is_goal_reached()
        truncated = self.current_step >= self.max_episode_steps
        
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
        
    def _apply_action(self, action: np.ndarray):
        """Apply joint position increments."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        for i, actuator_id in enumerate(self.robot_actuators):
            if i < len(action):
                joint_id = self.robot_joints[i]
                current_pos = self.data.qpos[joint_id]
                target_pos = current_pos + action[i]
                
                # Apply joint limits
                joint_range = self.model.jnt_range[joint_id]
                if joint_range[0] < joint_range[1]:
                    target_pos = np.clip(target_pos, joint_range[0], joint_range[1])
                
                # Set control target
                self.data.ctrl[actuator_id] = target_pos
                
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs_components = []
        
        # Joint positions (7)
        for joint_id in self.robot_joints:
            obs_components.append(self.data.qpos[joint_id])
        
        # Joint velocities (7)
        for joint_id in self.robot_joints:
            obs_components.append(self.data.qvel[joint_id])
        
        # End-effector position (3)
        ee_pos = self.data.site_xpos[self.ee_site_id]
        obs_components.extend(ee_pos)
        
        # Target sphere position (3)
        obs_components.extend(self.current_target)
        
        # Distance to target (1)
        distance = np.linalg.norm(ee_pos - self.current_target)
        obs_components.append(distance)
        
        # Direction vector to target (normalized) (3)
        direction = self.current_target - ee_pos
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-6:
            direction = direction / direction_norm
        obs_components.extend(direction)
        
        return np.array(obs_components, dtype=np.float32)
        
    def _calculate_reward(self, action: np.ndarray) -> float:
        """
        Calculate reward for reaching the target sphere.
        
        Reward structure:
        - Dense distance-based reward for guidance
        - Success bonus for reaching target
        - Efficiency bonus for smooth motion
        - Penalty for excessive motion
        """
        ee_pos = self.data.site_xpos[self.ee_site_id]
        current_distance = np.linalg.norm(ee_pos - self.current_target)
        
        reward = 0.0
        
        # 1. Distance-based reward (dense guidance)
        if self.prev_distance is not None:
            progress = self.prev_distance - current_distance
            reward += progress * 10.0  # Scale progress reward
            
        # 2. Success bonus
        if current_distance <= self.goal_tolerance:
            reward += 100.0  # Large success bonus
            
        # 3. Proximity bonus (gets stronger as we get closer)
        proximity_bonus = np.exp(-current_distance * 5.0)
        reward += proximity_bonus
        
        # 4. Efficiency penalty (penalize large actions)
        action_penalty = np.sum(np.abs(action)) * 0.1
        reward -= action_penalty
        
        # 5. Velocity penalty (encourage smooth motion)
        joint_vels = np.array([self.data.qvel[joint_id] for joint_id in self.robot_joints])
        velocity_penalty = np.sum(np.abs(joint_vels)) * 0.05
        reward -= velocity_penalty
        
        # Update previous distance
        self.prev_distance = current_distance
        
        return reward
        
    def _is_goal_reached(self) -> bool:
        """Check if the target sphere has been reached."""
        ee_pos = self.data.site_xpos[self.ee_site_id]
        distance = np.linalg.norm(ee_pos - self.current_target)
        return distance <= self.goal_tolerance
        
    def _get_info(self) -> Dict:
        """Get environment info."""
        ee_pos = self.data.site_xpos[self.ee_site_id]
        distance = np.linalg.norm(ee_pos - self.current_target)
        
        return {
            'episode_step': self.current_step,
            'episode_return': self.episode_return,
            'distance_to_target': distance,
            'success': self._is_goal_reached(),
            'target_position': self.current_target.copy(),
            'ee_position': ee_pos.copy(),
            'initial_distance': self.initial_distance,
        }
        
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # Set good viewing angle
                self.viewer.cam.distance = 1.5
                self.viewer.cam.azimuth = 135
                self.viewer.cam.elevation = -15
                self.viewer.cam.lookat[0] = 0.4
                self.viewer.cam.lookat[1] = 0.0  
                self.viewer.cam.lookat[2] = 0.3
            
            self.viewer.sync()
            
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Test function for the environment
def test_environment():
    """Test the Franka RL environment with random actions."""
    print("Testing Franka RL Environment...")
    
    env = FrankaRLEnvironment(render_mode="human")
    
    for episode in range(3):
        obs, info = env.reset()
        print(f"\nEpisode {episode + 1}")
        print(f"Target: {info['target_position']}")
        print(f"Initial distance: {info['initial_distance']:.3f}m")
        
        episode_reward = 0
        for step in range(100):
            # Random action for testing
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            env.render()
            time.sleep(0.02)
            
            if terminated:
                print(f"SUCCESS! Reached target in {step + 1} steps")
                break
                
            if truncated:
                print(f"Episode truncated after {step + 1} steps")
                break
                
        print(f"Episode reward: {episode_reward:.2f}")
        print(f"Final distance: {info['distance_to_target']:.3f}m")
        
    env.close()
    print("Test complete!")


if __name__ == "__main__":
    test_environment()
