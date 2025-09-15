# Franka RL - Sphere Reaching with PPO

A reinforcement learning environment where a Franka Panda robot learns to reach randomly positioned spheres using Proximal Policy Optimization (PPO). The system demonstrates how robots can learn complex kinematic behaviors through trial and error, using physics simulation and neural networks.

## Overview

This implementation trains a 7-DOF Franka Panda robot to:

1. **Reach random target spheres**: Targets are generated within the robot's reachable workspace
2. **Learn efficient kinematics**: PPO discovers joint coordination patterns for accurate reaching
3. **Adapt to varying targets**: The policy generalizes to new sphere positions without retraining

## Mathematical Framework

### Robot Kinematics
The Franka Panda has 7 revolute joints with configuration vector **q** ∈ ℝ⁷. The forward kinematics map joint angles to end-effector position:

**p** = f(**q**) ∈ ℝ³

Where **p** is the 3D Cartesian position of the end-effector site.

### State Space
The observation vector **s** ∈ ℝ²⁴ includes:
- Joint positions: **q** ∈ ℝ⁷  
- Joint velocities: **q̇** ∈ ℝ⁷
- End-effector position: **p** ∈ ℝ³
- Target position: **p_target** ∈ ℝ³
- Distance to target: ||**p** - **p_target**|| ∈ ℝ
- Direction vector: (**p_target** - **p**) / ||**p_target** - **p**|| ∈ ℝ³

### Action Space
Actions **a** ∈ ℝ⁷ represent joint position increments:
**q_target** = **q_current** + **a**

Actions are clipped to [-0.1, 0.1] radians per timestep for stability and safety.

### Reward Function
The reward function balances reaching accuracy with control efficiency:

R(**s**, **a**) = -||**p** - **p_target**|| - λ||**a**||₁ + R_success

Where:
- Primary term: negative Euclidean distance (promotes proximity)
- Action penalty: λ = 0.01 (encourages smooth motion)  
- Success bonus: R_success = 10.0 when ||**p** - **p_target**|| < 0.025m

## Physics Simulation

### MuJoCo Integration
The environment uses MuJoCo physics engine with:
- Position actuators with stiffness kp = 300 N⋅m/rad
- Joint limits enforced from Franka specifications
- 500Hz simulation timestep with 40Hz control frequency
- Collision detection disabled for training efficiency

### Workspace Constraints
Target spheres are constrained to reachable regions:
- X: [0.2, 0.7] m (forward reach)
- Y: [-0.4, 0.4] m (lateral reach)  
- Z: [0.1, 0.6] m (vertical reach)

This ensures all targets lie within the robot's kinematic workspace, preventing unreachable goals.

## PPO Training Algorithm

### Policy Architecture
The neural network policy uses:
- **Network**: 2 hidden layers with 256 units each
- **Activation**: ReLU activation functions
- **Input**: 24-dimensional observation vector
- **Output**: 7-dimensional action vector (joint increments)

### PPO Hyperparameters
Optimized for robotic manipulation tasks:
- **Learning rate**: 3×10⁻⁴ with Adam optimizer
- **Rollout length**: 2048 timesteps per environment
- **Batch size**: 64 samples per gradient update
- **Training epochs**: 10 epochs per rollout
- **Discount factor**: γ = 0.99
- **GAE parameter**: λ = 0.95 
- **Clipping range**: ε = 0.2
- **Entropy coefficient**: 0.01 (exploration bonus)

### Training Dynamics
The learning process involves:
1. **Exploration**: Random actions initially explore the workspace
2. **Credit assignment**: Temporal difference learning assigns rewards to action sequences
3. **Policy gradient**: PPO updates the neural network to increase expected reward
4. **Generalization**: The learned policy handles new target positions

## Usage

### 1. Test Robot Movement
```bash
mjpython test_robot_movement.py
```

Verifies that:
- Robot has functional actuators
- Actions produce joint movements
- End-effector responds to control signals
- Distance to target changes over time

### 2. Demo the Environment
```bash
mjpython demo_franka_rl.py
```

Interactive demonstrations:
- Random actions with visual feedback
- Workspace boundary exploration  
- Observation space structure analysis

### 3. Train with PPO
```bash
python train_franka_rl.py
```

Training configuration:
- 4 parallel environments for sample efficiency
- Automatic checkpointing every 10,000 timesteps
- Model evaluation every 5,000 timesteps
- Tensorboard logging for progress tracking
- Success rate monitoring during training

### 4. Monitor Training Progress
```bash
tensorboard --logdir logs/franka_rl
```

Key metrics tracked:
- Episode rewards (learning progress)
- Success rates (task completion)
- Episode lengths (efficiency improvement)
- Policy entropy (exploration vs exploitation)

### 5. Evaluate Trained Policy
```bash
mjpython evaluate_trained_model.py
```

Performance analysis:
- Success rate across diverse targets
- Trajectory efficiency measurements
- Response time to new configurations
- Visual demonstration of learned behaviors

## Implementation Details

### Actuator Discovery
The environment automatically detects available actuators in the MuJoCo model:
- Searches for arm joint actuators using naming patterns
- Handles models with additional actuators (e.g., gripper)
- Provides runtime validation of control authority

### Control Safety
Multiple safety mechanisms prevent dangerous behaviors:
- Joint position limits enforced from robot specifications
- Action clipping to [-0.1, 0.1] rad/timestep maximum
- Graceful handling of non-arm actuators
- Physics timestep validation

### Code Structure
- `franka_rl.py`: Core RL environment with MuJoCo integration
- `train_franka_rl.py`: PPO training with parallel environments
- `demo_franka_rl.py`: Interactive environment demonstration  
- `evaluate_trained_model.py`: Trained policy performance analysis
- `test_robot_movement.py`: Actuator and movement validation

## Mathematical Insights

### Learned Inverse Kinematics
The neural network policy effectively learns an approximate inverse kinematic mapping:

**q̇** ≈ π(**s**) where **s** includes (**p_target** - **p**)

This allows the robot to generate appropriate joint velocity patterns for reaching without explicit analytical solutions.

### Generalization Properties  
The policy demonstrates generalization across:
- **Spatial positions**: Reaching targets throughout the workspace
- **Initial configurations**: Success from various starting joint angles
- **Temporal dynamics**: Adapting trajectory length based on distance

### Convergence Characteristics
Training typically exhibits:
- **Phase 1** (0-50k steps): Random exploration, low success rate
- **Phase 2** (50k-200k steps): Rapid improvement, learning basic reaching
- **Phase 3** (200k+ steps): Refinement of efficiency and precision

The learned policy encodes complex sensorimotor coordination that would be challenging to program analytically, demonstrating the power of reinforcement learning for robotic control tasks.
