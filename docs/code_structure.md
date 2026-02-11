# MolmoSpaces

A modular robotics simulation framework built on MuJoCo for data generation and robot control experiments.

## Code Structure

```
molmo_spaces/
├── config/                          # Configuration management
│   ├── abstract_config.py          # Base configuration class
│   └── abstract_exp_config.py      # Experiment configuration interface
├── controllers/                     # Robot control interfaces
│   ├── abstract.py                 # Base controller class
│   ├── base_pose.py                # Base pose controller
│   ├── holo_base_pose.py           # Holonomic base controller
│   ├── joint_pos.py                # Joint position controller
│   └── joint_vel.py                # Joint velocity controller
├── data_generation/                 # Data generation pipeline
│   ├── config/                     # Data generation configs
│   │   └── rby1_door_opening_config.py
│   ├── main.py                     # Entry point for data generation
│   └── pipeline.py                 # Parallel rollout runner
├── env/                            # Environment abstractions
│   ├── data_views.py               # Data view utilities
│   └── env.py                      # MuJoCo environment wrapper
├── kinematics/                     # Kinematic solvers
│   ├── mujoco_kinematics.py        # MuJoCo-based kinematics
│   └── rby1_kinematics.py          # RBY1-specific kinematics
├── planner/                        # Motion planning
│   ├── abstract.py                 # Base planner interface
│   └── curobo_planner.py           # CuRobo integration
├── policy/                         # Policy implementations
│   ├── abstract.py                 # Base policy interface
│   ├── learned_policy/             # Learned policies
│   ├── planner_policy/             # Planning-based policies
│   └── teleop_policy/              # Teleoperation policies
├── robots/                         # Robot implementations
│   ├── abstract.py                 # Base robot interface
│   ├── rby1.py                     # RBY1 robot implementation
│   └── robot_views/                # Robot view abstractions
└── tasks/                          # Task definitions
    ├── robot_specific/             # Robot-specific tasks
    ├── task_sampler.py             # Task sampling interface
    └── task.py                     # Base task interface
```

## Information Flow

### Data Generation Pipeline

1. **Entry Point**: `main.py` loads experiment configuration and initializes the pipeline.
2. **Configuration**: Experiment configs inherit from `MlSpacesExpConfig` and define:
    - Task sampler configuration
    - Robot configuration
    - Policy configuration
    - Fixed task parameters
3. **Pipeline**: `ParallelRolloutRunner` manages parallel execution:
    - Creates task samplers for each worker thread
    - Samples tasks with fixed parameters
    - Initializes policies and environments
    - Runs rollouts and collects data
4. **Task Execution**: Each episode follows:
    - Task sampling → Policy initialization → Episode rollout → Success evaluation

### Core Components

- **Tasks**: Define robot objectives and success criteria
- **Policies**: Generate actions from observations (planner, teleop, learned)
- **Robots**: Interface with MuJoCo simulation and provide control
- **Environments**: Manage MuJoCo models and parallel execution
- **Controllers**: Handle low-level robot control commands

## Configuration Hierarchy

The framework uses a hierarchical configuration system:

1. **Experiment Config** (`MlSpacesExpConfig`): Top-level experiment parameters
2. **Task Sampler Config**: Defines task sampling ranges and constraints
3. **Task Config**: Fixed parameters for specific task instances
4. **Robot Config**: Robot-specific settings and control modes
5. **Policy Config**: Policy type and parameters

## Example Running Command

```bash
# Set environment variables (MacOS)
export PYTHONPATH=${PYTHONPATH}:.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Run door opening data generation
python -m molmo_spaces.data_generation.main \
    molmo_spaces/data_generation/config/rby1_door_opening_config.py \
    DoorOpeningDataGenConfig
```

## Key Features

- **Modular Design**: Clean separation between tasks, policies, and robots
- **Parallel Execution**: Multi-threaded data generation with thread-safe counters
- **Flexible Configuration**: Hierarchical config system for easy experimentation
- **Multiple Policy Types**: Support for planning, teleoperation, and learned policies
- **Robot Agnostic**: Abstract interfaces allow easy robot integration
