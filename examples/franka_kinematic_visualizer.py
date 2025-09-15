#!/usr/bin/env python3
"""
Franka Kinematic Visualization Demo

Interactive demonstration of the Franka Panda robot reaching cube faces.
Users select target faces and watch the robot compute inverse kinematics,
plan a path, and execute the movement using MuJoCo simulation.

Features:
- Interactive face selection on a colored cube
- Real-time inverse kinematics solving
- Smooth trajectory planning and execution
- Educational visualization of robot kinematics

Requires: mujoco>=3.0.0, numpy, scipy, matplotlib
Note: Expects mujoco_menagerie repository at ~/code/mujoco_menagerie
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from scipy.spatial.transform import Rotation
from typing import List, Optional
import random
import math


class FrankaKinematicVisualizer:
    """
    Interactive kinematic demonstration using a Franka Panda robot.
    
    Sets up a MuJoCo simulation where users can select cube faces for the robot
    to touch. The robot computes inverse kinematics, plans a path, and executes
    the movement while displaying educational information about the process.
    """
    
    def __init__(self, menagerie_path: str = None):
        """
        Initialize the demo with robot model and simulation setup.
        
        Args:
            menagerie_path: Path to mujoco_menagerie repository
        """
        # Set default menagerie path if not provided
        if menagerie_path is None:
            import os
            self.menagerie_path = os.path.expanduser("~/code/mujoco_menagerie")
        else:
            self.menagerie_path = menagerie_path
        self.model = None
        self.data = None
        self.viewer = None
        
        # Cube setup
        self.cube_size = 0.05  # 5cm cube
        self.cube_position = np.array([0.5, 0.0, 0.05])  # Position on table surface
        
        # Face definitions (center positions relative to cube center)
        self.cube_faces = {
            'front': np.array([0.0, -self.cube_size, 0.0]),   # -Y face
            'back': np.array([0.0, self.cube_size, 0.0]),     # +Y face  
            'left': np.array([-self.cube_size, 0.0, 0.0]),    # -X face
            'right': np.array([self.cube_size, 0.0, 0.0]),    # +X face
            'top': np.array([0.0, 0.0, self.cube_size]),      # +Z face
            'bottom': np.array([0.0, 0.0, -self.cube_size])   # -Z face
        }
        
        # Face normals (pointing outward)
        self.face_normals = {
            'front': np.array([0.0, -1.0, 0.0]),
            'back': np.array([0.0, 1.0, 0.0]),
            'left': np.array([-1.0, 0.0, 0.0]),
            'right': np.array([1.0, 0.0, 0.0]),
            'top': np.array([0.0, 0.0, 1.0]),
            'bottom': np.array([0.0, 0.0, -1.0])
        }
        
        # Target tracking
        self.target_face = None
        self.target_position = None
        self.target_orientation = None
        
        # Path execution
        self.planned_path = []
        self.current_path_index = 0
        self.path_execution_active = False
        
        self._setup_simulation()
    
    def _setup_simulation(self):
        """
        Build the MuJoCo simulation with Franka robot and target cube.
        
        Loads the robot model, adds a cube with colored face markers,
        sets up lighting, and initializes the robot pose.
        """
        import os
        franka_model_path = os.path.join(self.menagerie_path, "franka_emika_panda", "panda.xml")
        
        with open(franka_model_path, 'r') as f:
            franka_xml = f.read()
        
        # Update mesh paths to absolute paths
        xml_content = franka_xml.replace(
            'meshdir="assets"',
            f'meshdir="{self.menagerie_path}/franka_emika_panda/assets"'
        )
        
        # Add tracking site to robot hand
        xml_content = xml_content.replace(
            '<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">',
            '''<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                  <site name="end_effector" pos="0 0 0.103" size="0.02" rgba="1 0 0 1"/>'''
        )
        
        # Add scene elements to the robot model
        xml_content = xml_content.replace('</worldbody>', f'''
            <!-- Scene lighting -->
            <light name="main" pos="0 0 3" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" directional="false"/>
            <light name="fill1" pos="1 1 2" diffuse="0.4 0.4 0.4" directional="false"/>
            <light name="fill2" pos="-1 -1 2" diffuse="0.4 0.4 0.4" directional="false"/>
            
            <!-- Floor -->
            <geom name="floor" type="plane" pos="0 0 0" size="3 3 0.1" rgba="0.95 0.95 0.95 1"/>
            
            <!-- Target cube -->
            <body name="target_cube" pos="{self.cube_position[0]} {self.cube_position[1]} {self.cube_position[2]}">
              <geom name="cube_main" type="box" size="{self.cube_size} {self.cube_size} {self.cube_size}" 
                    rgba="0.9 0.9 0.9 1" group="1"/>
              
              <!-- Face markers for identification -->
              <site name="face_front" pos="0 -{self.cube_size + 0.005} 0" size="0.015" rgba="1 0 0 1"/>
              <site name="face_back" pos="0 {self.cube_size + 0.005} 0" size="0.015" rgba="0 1 0 1"/>
              <site name="face_left" pos="-{self.cube_size + 0.005} 0 0" size="0.015" rgba="0 0 1 1"/>
              <site name="face_right" pos="{self.cube_size + 0.005} 0 0" size="0.015" rgba="1 1 0 1"/>
              <site name="face_top" pos="0 0 {self.cube_size + 0.005}" size="0.015" rgba="1 0 1 1"/>
              <site name="face_bottom" pos="0 0 -{self.cube_size + 0.005}" size="0.015" rgba="0 1 1 1"/>
            </body>
            
            <!-- Target indicator -->
            <body name="target_indicator" pos="0 0 -1">
              <geom name="target_sphere" type="sphere" size="0.02" rgba="1 0 0 0.8" pos="0 0 0"/>
            </body>
          </worldbody>''')
        
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.data = mujoco.MjData(self.model)
        
        self._set_neutral_pose()
        
        print("Simulation initialized")
        print(f"Robot model: {self.menagerie_path}")
        print(f"Cube at: {self.cube_position}")
    
    def _set_neutral_pose(self):
        """Set robot to starting pose suitable for table manipulation."""
        neutral_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        
        for i in range(7):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            if joint_id >= 0:
                self.data.qpos[joint_id] = neutral_qpos[i]
        
        mujoco.mj_forward(self.model, self.data)
    
    def get_user_face_selection(self) -> str:
        """
        Interactive face selection with input validation.
        
        Returns:
            Selected face name
        """
        print("\nCUBE FACE SELECTION")
        print("=" * 40)
        print("Available cube faces:")
        print("  1. front  (Red marker, -Y direction)")
        print("  2. back   (Green marker, +Y direction)")
        print("  3. left   (Blue marker, -X direction)")
        print("  4. right  (Yellow marker, +X direction)")
        print("  5. top    (Magenta marker, +Z direction)")
        print("  6. bottom (Cyan marker, -Z direction)")
        print()
        
        face_options = {
            '1': 'front', 'front': 'front', 'f': 'front',
            '2': 'back', 'back': 'back', 'b': 'back',
            '3': 'left', 'left': 'left', 'l': 'left',
            '4': 'right', 'right': 'right', 'r': 'right',
            '5': 'top', 'top': 'top', 't': 'top',
            '6': 'bottom', 'bottom': 'bottom', 'bot': 'bottom'
        }
        
        while True:
            try:
                user_input = input("Select cube face (1-6, name, or first letter): ").strip().lower()
                
                if user_input in face_options:
                    selected_face = face_options[user_input]
                    break
                else:
                    print("Invalid selection. Please try again.")
                    
            except KeyboardInterrupt:
                print("\nDemo ended by user")
                return None
        
        self.target_face = selected_face
        
        # Calculate target position and approach
        face_offset = self.cube_faces[self.target_face]
        self.target_position = self.cube_position + face_offset
        
        face_normal = self.face_normals[self.target_face]
        
        # Distance to approach each face (varies for reachability)
        face_distances = {
            'front': 0.06, 'back': 0.08, 'left': 0.06,
            'right': 0.06, 'top': 0.06, 'bottom': 0.08
        }
        
        approach_distance = face_distances[self.target_face]
        approach_position = self.target_position + approach_distance * face_normal
        self.target_position = approach_position
        
        # Calculate end-effector orientation (tool points toward face)
        z_axis = -face_normal  # Tool Z-axis points toward surface
        
        # Generate perpendicular axes
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross([0, 0, 1], z_axis)
        else:
            x_axis = np.cross([1, 0, 0], z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        self.target_orientation = Rotation.from_matrix(rotation_matrix).as_quat()
        
        print(f"\nSelected: {self.target_face.upper()} face")
        print(f"   Face center: {self.cube_position + self.cube_faces[self.target_face]}")
        print(f"   Approach position: {self.target_position}")
        print(f"   Face normal: {face_normal}")
        
        return self.target_face
    
    def compute_inverse_kinematics(self, target_pos: np.ndarray, target_quat: np.ndarray, 
                                 max_iterations: int = 100, tolerance: float = 1e-4) -> Optional[np.ndarray]:
        """
        Solve inverse kinematics using iterative Jacobian method.
        
        Uses damped least squares approach:
        - Compute pose error (position + orientation)
        - Calculate Jacobian for end-effector
        - Update joint angles to reduce error
        - Repeat until convergence
        
        Args:
            target_pos: Target position [x, y, z]
            target_quat: Target orientation quaternion [x, y, z, w]
            max_iterations: Maximum solver iterations
            tolerance: Position error tolerance
            
        Returns:
            Joint angles solution, or None if failed
        """
        print("\nSolving inverse kinematics...")
        print(f"   Target position: {target_pos}")
        print(f"   Target orientation: {target_quat}")
        
        # Save current state to restore after IK computation
        full_qpos_backup = self.data.qpos.copy()
        full_qvel_backup = self.data.qvel.copy()
        
        def vee(S):
            """Extract vector from skew-symmetric matrix"""
            return np.array([S[2,1]-S[1,2], S[0,2]-S[2,0], S[1,0]-S[0,1]]) * 0.5
        
        def ori_error(R, R_d):
            """Compute orientation error between current and desired rotation"""
            return 0.5 * vee(R_d.T @ R - R.T @ R_d)
        
        target_rot = Rotation.from_quat(target_quat)
        R_d = target_rot.as_matrix()
        
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        if ee_site_id < 0:
            print("   Error: end_effector site not found")
            return None
        
        # Starting poses optimized for each face direction
        face_start_configs = {
            'front': np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
            'back': np.array([3.14, -0.5, 0.0, -2.0, 0.0, 1.8, 0.785]),
            'left': np.array([-1.57, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
            'right': np.array([1.57, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
            'top': np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
            'bottom': np.array([0.0, 0.5, 0.0, -1.5, 0.0, 2.0, 0.785])
        }
        
        target_face_name = getattr(self, 'target_face', 'front')
        
        if target_face_name in face_start_configs:
            start_config = face_start_configs[target_face_name]
            print(f"   Using {target_face_name}-optimized starting pose")
        else:
            start_config = face_start_configs['front']
            print(f"   Using default starting pose")
            
        self.data.qpos[:7] = start_config
        self.data.qvel[:] = 0
        
        # IK solver parameters
        alpha = 0.2      # step size
        lam = 1e-2       # damping factor
        max_iterations = 300
        
        q_lo = self.model.jnt_range[:7, 0]
        q_hi = self.model.jnt_range[:7, 1]
        
        for iteration in range(max_iterations):
            mujoco.mj_forward(self.model, self.data)
            
            # Current end-effector pose
            p = self.data.site_xpos[ee_site_id].copy()
            R = self.data.site_xmat[ee_site_id].reshape(3, 3).copy()
            
            # Calculate pose errors
            e_p = target_pos - p
            e_R = ori_error(R, R_d)
            
            # Weight position higher than orientation
            w_pos, w_rot = 1.0, 0.3
            
            # Limit rotation step size for stability
            max_rot_step = 0.2
            rn = np.linalg.norm(e_R) + 1e-9
            if rn > max_rot_step:
                e_R = e_R * (max_rot_step / rn)
            
            e = np.hstack([w_pos * e_p, w_rot * e_R])
            error_norm = np.linalg.norm(e)
            pos_error_norm = np.linalg.norm(e_p)
            
            # Check for convergence
            if error_norm < tolerance:
                print(f"   Converged in {iteration + 1} iterations")
                print(f"   Final error: {error_norm:.6f}")
                
                solution = self.data.qpos[:7].copy()
                self.data.qpos[:] = full_qpos_backup
                self.data.qvel[:] = full_qvel_backup
                mujoco.mj_forward(self.model, self.data)
                return solution
            
            # Accept good position accuracy even with orientation error
            if pos_error_norm < 0.02:
                print(f"   Position converged in {iteration + 1} iterations")
                print(f"   Position error: {pos_error_norm:.6f}m")
                
                solution = self.data.qpos[:7].copy()
                self.data.qpos[:] = full_qpos_backup
                self.data.qvel[:] = full_qvel_backup
                mujoco.mj_forward(self.model, self.data)
                return solution
            
            # Compute Jacobian for end-effector
            Jp = np.zeros((3, self.model.nv))
            Jr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, Jp, Jr, ee_site_id)
            J = np.vstack([Jp[:, :7], Jr[:, :7]])
            
            # Damped least squares solution
            JJt = J @ J.T
            A = JJt + (lam**2) * np.eye(6)
            try:
                y = np.linalg.solve(A, e)
                dq = J.T @ y
                
                # Add null-space motion to avoid joint limits
                q_mid = 0.5 * (q_lo + q_hi)
                dq_null = 0.05 * (q_mid - self.data.qpos[:7])
                
                JsharpJ = J.T @ np.linalg.solve(JJt + (lam**2) * np.eye(6), J)
                N = np.eye(7) - JsharpJ
                dq += N @ dq_null
                
            except np.linalg.LinAlgError:
                print(f"   Singular matrix at iteration {iteration + 1}")
                lam *= 10
                continue
            
            # Update joint positions
            self.data.qpos[:7] += alpha * dq
            self.data.qpos[:7] = np.clip(self.data.qpos[:7], q_lo, q_hi)
            
            # Progress updates
            if (iteration + 1) % 20 == 0:
                print(f"   Iteration {iteration + 1}: error = {error_norm:.6f}, pos = {pos_error_norm:.6f}m")
        
        # Final convergence check
        mujoco.mj_forward(self.model, self.data)
        final_pos = self.data.site_xpos[ee_site_id].copy()
        final_pos_error = np.linalg.norm(target_pos - final_pos)
        
        print(f"   Did not converge in {max_iterations} iterations")
        print(f"   Final position error: {final_pos_error:.6f}m")
        
        # Accept if reasonably close
        if final_pos_error < 0.12:
            print(f"   Accepting solution (position error acceptable)")
            solution = self.data.qpos[:7].copy()
        else:
            print(f"   Rejecting solution (position error too large)")
            solution = None
            
        # Restore original state
        self.data.qpos[:] = full_qpos_backup
        self.data.qvel[:] = full_qvel_backup
        mujoco.mj_forward(self.model, self.data)
        
        return solution
    
    def plan_path(self, target_qpos: np.ndarray, num_waypoints: int = 10) -> List[np.ndarray]:
        """
        Plan smooth path from current to target joint positions.
        
        Uses linear interpolation in joint space for simplicity.
        More sophisticated methods (RRT, splines) could be used for
        real applications requiring obstacle avoidance.
        
        Args:
            target_qpos: Target joint positions
            num_waypoints: Number of intermediate waypoints
            
        Returns:
            List of waypoint joint positions
        """
        print(f"\nPlanning path with {num_waypoints} waypoints...")
        
        current_qpos = self.data.qpos[:7].copy()
        
        # Linear interpolation in joint space
        path = []
        for i in range(num_waypoints + 1):
            alpha = i / num_waypoints
            waypoint = (1 - alpha) * current_qpos + alpha * target_qpos
            path.append(waypoint)
        
        joint_distance = np.linalg.norm(target_qpos - current_qpos)
        print(f"   Path planned: {len(path)} waypoints")
        print(f"   Joint space distance: {joint_distance:.3f} rad")
        
        self.planned_path = path
        self.current_path_index = 0
        
        return path
    
    def visualize_path(self):
        """
        Update target visualization to show robot destination.
        """
        if not self.planned_path:
            return
        
        print(f"\nUpdating target visualization...")
        
        # Position target indicator at destination
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_indicator")
        if target_body_id >= 0:
            self.model.body_pos[target_body_id] = self.target_position
        
        print(f"   Robot will approach {self.target_face} face")
    
    def execute_path_step(self, step_count: int = 0) -> bool:
        """
        Execute one step of the planned path.
        
        Sends target joint positions to actuators and checks for
        waypoint completion based on position error.
        
        Returns:
            True if path continues, False if complete
        """
        verbose = (step_count % 50 == 1) or (step_count <= 10)
        
        if not self.planned_path or self.current_path_index >= len(self.planned_path):
            return False
        
        # Get target and current positions
        target_qpos = self.planned_path[self.current_path_index]
        current_qpos = self.data.qpos[:7].copy()
        current_qvel = self.data.qvel[:7].copy()
        
        # Send target positions to actuators
        pos_error = target_qpos - current_qpos
        pos_error_norm = np.linalg.norm(pos_error)
        
        if verbose:
            print(f"   Waypoint {self.current_path_index+1}/{len(self.planned_path)}: error {pos_error_norm:.4f}")
        
        self.data.ctrl[:7] = target_qpos
            
        # Clamp to actuator limits
        if self.model.actuator_ctrlrange.shape[0] >= 7:
            lo = self.model.actuator_ctrlrange[:7, 0]
            hi = self.model.actuator_ctrlrange[:7, 1]
            self.data.ctrl[:7] = np.clip(self.data.ctrl[:7], lo, hi)
        
        tolerance = 0.03  # ~1.7 degrees
        
        # Check waypoint completion
        vel_norm = np.linalg.norm(current_qvel)
        
        if pos_error_norm < tolerance:
            self.current_path_index += 1
            if self.current_path_index < len(self.planned_path):
                print(f"   Reached waypoint {self.current_path_index}/{len(self.planned_path)}")
            else:
                print(f"   Path execution complete!")
                return False
        elif pos_error_norm < tolerance * 2 and vel_norm < 0.1:
            print(f"   Close enough - advancing (error: {pos_error_norm:.4f})")
            self.current_path_index += 1
            if self.current_path_index >= len(self.planned_path):
                print(f"   Path execution complete!")
                return False
        
        return True
    
    def simulate_touch(self, touch_duration: float = 2.0):
        """
        Simulate robot touching the target face.
        
        Args:
            touch_duration: How long to hold position (seconds)
        """
        print(f"\nSimulating touch on {self.target_face} face...")
        
        # Calculate contact position
        face_normal = self.face_normals[self.target_face]
        face_center = self.cube_position + self.cube_faces[self.target_face]
        touch_position = face_center - 0.01 * face_normal
        
        # Check current position
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        current_pos = self.data.site_xpos[ee_site_id].copy()
        distance_to_touch = np.linalg.norm(touch_position - current_pos)
        
        if distance_to_touch < 0.05:
            print(f"   Contact achieved! Distance: {distance_to_touch:.3f}m")
        else:
            print(f"   Near target. Distance: {distance_to_touch:.3f}m")
        
        start_time = time.time()
        print(f"   Holding position for {touch_duration}s...")
        
        while time.time() - start_time < touch_duration:
            # Maintain current joint positions
            mujoco.mj_step(self.model, self.data)
            
            if self.viewer:
                self.viewer.sync()
            
            time.sleep(0.01)
        
        print(f"   Touch complete!")
    
    def reset_to_neutral(self):
        """
        Smoothly return robot to neutral starting pose.
        """
        print(f"\nResetting to neutral pose...")
        
        current_qpos = self.data.qpos[:7].copy()
        target_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        
        # Smooth interpolation back to neutral
        num_steps = 50
        for i in range(num_steps + 1):
            alpha = i / num_steps
            waypoint = (1 - alpha) * current_qpos + alpha * target_qpos
            self.data.qpos[:7] = waypoint
            
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)
                if self.viewer:
                    self.viewer.sync()
                time.sleep(0.005)
        
        self.data.qpos[:7] = target_qpos
        mujoco.mj_forward(self.model, self.data)
        
        # Clear target state
        self.target_face = None
        self.target_position = None
        self.target_orientation = None
        self.planned_path = []
        self.current_path_index = 0
        self.path_execution_active = False
        
        print(f"   Reset complete!")
    
    def highlight_target_face(self):
        """
        Show which face is targeted (relies on colored markers).
        """
        if self.target_face:
            print(f"   Target: {self.target_face} face (see colored markers)")
    
    def print_kinematic_info(self):
        """
        Display kinematic analysis of current state.
        """
        if self.target_position is None:
            return
        
        print(f"\nKINEMATIC ANALYSIS")
        print(f"=" * 50)
        
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        current_pos = self.data.site_xpos[ee_site_id]
        
        print(f"End-Effector: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
        print(f"Target: [{self.target_position[0]:.3f}, {self.target_position[1]:.3f}, {self.target_position[2]:.3f}]")
        print(f"Error: {np.linalg.norm(self.target_position - current_pos):.3f}m")
        
        # Workspace check
        distance_to_target = np.linalg.norm(self.target_position - np.array([0, 0, 0.333]))
        reachable = "Yes" if distance_to_target < 0.85 else "No"
        print(f"Distance from base: {distance_to_target:.3f}m (reachable: {reachable})")
        
    def run_demonstration(self):
        """
        Run the interactive demonstration.
        
        Complete workflow:
        1. User selects target cube face
        2. Compute inverse kinematics
        3. Plan and execute path
        4. Simulate touch
        5. Reset for next iteration
        """
        print("\n" + "="*60)
        print("FRANKA KINEMATIC DEMONSTRATION")
        print("="*60)
        print("Select cube faces for the robot to reach.")
        print("Watch the kinematics computation, path planning,")
        print("and smooth trajectory execution.")
        print("="*60)
        
        # Start the viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer
            
            # Set camera for good viewing angle
            viewer.cam.distance = 1.2
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -15
            viewer.cam.lookat[0] = 0.3
            viewer.cam.lookat[1] = 0.0  
            viewer.cam.lookat[2] = 0.3
            
            print("\nCube faces are marked with colors:")
            print("   Red=Front, Green=Back, Blue=Left, Yellow=Right, Magenta=Top, Cyan=Bottom")
            print("\nPress ENTER to begin...")
            input()
            
            while True:
                try:
                    # Step 1: Select target
                    self.get_user_face_selection()
                    self.highlight_target_face()
                    
                    # Step 2: Solve inverse kinematics
                    print(f"\nComputing kinematics for {self.target_face} face...")
                    
                    target_qpos = self.compute_inverse_kinematics(
                        self.target_position, 
                        self.target_orientation
                    )
                    
                    if target_qpos is None:
                        print("Could not solve IK for this target.")
                        print("Target may be outside workspace. Try another face.\n")
                        continue
                    
                    # Step 3: Plan path
                    self.plan_path(target_qpos)
                    self.visualize_path()
                    
                    # Step 4: Execute movement
                    print(f"\nExecuting movement to {self.target_face} face...")
                    
                    self.path_execution_active = True
                    step_count = 0
                    
                    while self.path_execution_active:
                        step_count += 1
                        
                        continuing = self.execute_path_step(step_count)
                        
                        if not continuing:
                            self.path_execution_active = False
                        
                        mujoco.mj_step(self.model, self.data)
                        viewer.sync()
                        time.sleep(0.005)
                        
                        # Safety limit
                        if step_count > 5000:
                            print(f"   Stopping after {step_count} steps (safety limit)")
                            self.path_execution_active = False
                    
                    # Step 5: Touch simulation
                    self.simulate_touch(touch_duration=2.0)
                    
                    # Step 6: Final analysis
                    self.print_kinematic_info()
                    
                    # Step 7: Reset
                    self.reset_to_neutral()
                    
                    print("\n" + "="*50)
                    print("Task complete! Ready for next face.")
                    print("="*50)
                    
                except KeyboardInterrupt:
                    print("\n\nDemo ended by user")
                    break
                except Exception as e:
                    print(f"\nError: {e}")
                    print("Resetting robot and continuing...")
                    try:
                        self.reset_to_neutral()
                    except:
                        print("Reset failed. Ending demo.")
                        break


def main():
    """
    Main entry point for the kinematic demonstration.
    """
    try:
        visualizer = FrankaKinematicVisualizer()
        visualizer.run_demonstration()
        
    except Exception as e:
        print(f"Initialization failed: {e}")
        print("\nRequirements:")
        print("1. MuJoCo 3.0+ installed")
        print("2. mujoco_menagerie at ~/code/mujoco_menagerie")
        print("3. Dependencies: numpy, scipy, matplotlib")


if __name__ == "__main__":
    main()
