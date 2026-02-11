# Task Samplers

This directory contains task implementations and task samplers that handle scene randomization and task configuration.

## Move Task Sampler

The `move_task_sampler.py` and `move_tasks.py` files contain a complete implementation of a pick task sampler that was extracted from the reference datagen script functionality.

### Key Features Extracted

The task sampler implements the following scene randomization capabilities that were previously handled by the external `sim2real` module:

#### 1. **Intelligent Table-Based Object Placement**
- **Table Detection**: Automatically detects tables in the scene by name keywords and geometry
- **Surface Placement**: Objects are placed on detected table surfaces with proper height
- **Collision Avoidance**: Maintains minimum separation between objects on table
- **Boundary Respect**: Objects placed within table boundaries with configurable margins
- **Orientation Randomization**: Yaw-based rotation while keeping objects stable on surface

#### 2. **Smart Robot Perimeter Placement**
- **Perimeter Sampling**: Robot positioned around table perimeter at configurable distances
- **Table-Facing Orientation**: Robot automatically oriented to face the table
- **Clearance Checking**: Ensures minimum clearance from table corners
- **Ground Placement**: Robot base positioned on ground level with height offset support

#### 3. **Reachability Analysis**
- **Distance Checking**: Ensures objects are within robot's reach envelope
- **Workspace Validation**: Configurable minimum/maximum reach distances
- **Re-sampling Logic**: Automatically re-samples positions if objects are unreachable
- **Fallback Support**: Falls back to manual placement if reachability constraints can't be met

#### 4. **Camera and Rendering Configuration**
- **Image Resolution Sampling**: Multiple resolution options for data collection
- **Camera Setup**: Automatic detection and configuration of scene cameras
- **Rendering Parameters**: Configurable rendering settings for data generation

#### 5. **Task Parameters**
- **Success Criteria**: Configurable position and rotation thresholds
- **Control Timing**: Configurable control periods and simulation time steps
- **Speed Settings**: Robot motion speed configuration (slow/fast modes)
- **Task Horizon**: Maximum episode length configuration

### Architecture Integration

The implementation follows the established pattern in the codebase:

```
MoveToPoseTaskSamplerConfig  # Configuration for randomization parameters
    ↓
MoveToPoseTaskSampler       # Implements scene setup and randomization
    ↓
MoveToPoseTaskConfig        # Task-specific configuration
    ↓
MoveToPoseTask              # Task implementation with observations/rewards
```

### Usage Example

```python
# In your experiment configuration:
def get_task_sampler_config(self):
    return (MoveToPoseTaskSampler, MoveToPoseTaskSamplerConfig(
        # Object randomization
        pickup_obj_names=["cube", "sphere", "cylinder"],
        place_target_names=["target_table", "target_shelf"],

        # NEW: Intelligent table-based placement
        table_detection_keywords=["table", "desk", "surface"],
        use_table_based_placement=True,

        # Robot placement around table perimeter
        robot_distance_from_table=(0.8, 1.2),  # Distance from table
        min_robot_clearance=0.3,  # Clearance from corners

        # Object placement on table surface
        table_surface_margin=0.1,  # Margin from edges
        object_height_offset=0.05,  # Height above surface
        min_object_separation=0.15,  # Separation between objects

        # Reachability constraints
        max_reach_distance=0.9,  # Maximum reach
        min_reach_distance=0.3,  # Minimum reach
        check_reachability=True,  # Enable checking

        # Fallback parameters (if table detection fails)
        robot_base_positions=[(0.0, 0.0, 0.0)],
        pickup_obj_position_ranges=[((0.3, -0.3, 0.8), (0.7, 0.3, 1.0))],

        # Success criteria
        succ_pos_thresholds=[0.05],
        succ_rot_thresholds=[0.2],
    ))
```

### Comparison with Reference Script

| Reference Script Function | Extracted Implementation | Enhancement |
|---------------------------|--------------------------|-------------|
| `create_scene()` | `AbstractMujocoTaskSampler._load_scene()` | Same functionality |
| `setup_scene()` | `MoveToPoseTaskSampler._setup_scene_randomization()` | **Enhanced with table detection** |
| `SceneConfig` parameters | `MoveToPoseTaskSamplerConfig` class | **More comprehensive config** |
| Object position randomization | `_sample_object_position_on_table()` | **Table-aware placement** |
| Robot placement randomization | `_sample_robot_position_around_table()` | **Perimeter-based placement** |
| Success checking | `MoveToPoseTask._check_success_single()` | Same functionality |
| Camera configuration | `img_resolutions` parameter | Same functionality |
| *(New)* Table detection | `_detect_tables()` | **Automatic scene understanding** |
| *(New)* Reachability checking | `_check_reachability()` | **Workspace validation** |

### Benefits of Extraction

1. **Modularity**: Scene randomization is now properly encapsulated in task samplers
2. **Configurability**: All randomization parameters are configurable through config classes
3. **Reusability**: Task sampler can be used across different experiments and policies
4. **Maintainability**: Clear separation of concerns between scene setup, task logic, and data generation
5. **Extensibility**: Easy to add new randomization parameters or object types
6. **Integration**: Follows existing framework patterns and interfaces
7. **Intelligence**: **NEW** - Automatic table detection and scene understanding
8. **Realism**: **NEW** - Objects placed realistically on table surfaces within robot reach
9. **Robustness**: **NEW** - Fallback mechanisms when intelligent placement fails
10. **Efficiency**: **NEW** - Reduces failed episodes by ensuring reachable object placements

### Next Steps

To complete the integration, you would need to:

1. Implement the specific robot class (e.g., `FrankaFR3`)
2. Implement the policy class (e.g., heuristic controller from reference script)
3. Create scene XML files with the required object names
4. Test the task sampler with the data generation pipeline

The extracted functionality provides a solid foundation for modular, configurable scene randomization that can replace the external dependencies while maintaining all the randomization capabilities of the original reference script.
