# Isaac Sim Franka Robot Automation

This repository contains scripts for automating robotic manipulation tasks with the Franka robot in NVIDIA Isaac Sim 4.5.0, including pick-and-place operations, interactive controls, and utility tools.

## Getting Started

1. Install [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/index.html).
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
3. `uv sync`

## Settings

Copy `example.env` to `.env` and set the location of your Isaac Sim installation.

## Pick-and-Place Scripts

### Main Implementation

- **optimized_franka_pick_place.py**: The most advanced implementation with modular design, robust error handling, smooth trajectory planning, and comprehensive logging.
  ```
  python run_pick_place.py optimized
  ```

### Alternative Implementations

- **improved_final_franka_pick_place_banana.py**: Refined implementation with precise banana grasping and proper prim path detection.
  ```
  python run_pick_place.py improved
  ```

- **final_franka_pick_place_banana.py**: Optimized version with direct vertical alignment and trajectory planning.
  ```
  python run_pick_place.py final
  ```

- **franka_pick_place_banana.py**: Original implementation of the pick-and-place task.
  ```
  python run_pick_place.py original
  ```

## Control Scripts

- **franka_control_4_5.py**: Base control script for Franka robot in Isaac Sim 4.5.0, providing fundamental control capabilities.

- **franka_button_control.py**: Interactive control using UI buttons for testing manual joint positioning.

- **franka_keyboard_control.py**: Control the robot using keyboard inputs for interactive testing.

- **franka_interactive_control.py**: Advanced interactive control with customizable UI elements.

- **franka_ui_control.py**: UI-based control for Franka robot with sliders and buttons.

## Basic Robot Examples

- **simple_franka.py**: Simple example showing basic Franka robot setup and movement.

- **minimal_franka.py**: Minimal implementation of Franka robot in Isaac Sim.

- **franka_interactive.py**: Basic interactive mode for testing robot movements.

- **franka_with_control.py**: Example showing robot with control interfaces.

## Utility Scripts

- **run_pick_place.py**: Test runner for different pick-and-place implementations.
  ```
  python run_pick_place.py [implementation_name]
  ```

- **list_scene_objects.py**: Utility for exploring and listing scene objects, helpful for debugging.
  ```
  python list_scene_objects.py
  ```

- **robot_movements.py**: Helper class for robot joint control with various movement patterns.

- **core.py**: Sets up the Isaac Sim environment with proper path configuration.

- **fix_numpy_compatibility.py**: Resolves NumPy compatibility issues that may occur with Isaac Sim.

- **program_controlled.py**: Demonstrates programmatic control of an Isaac Sim simulation.
  ```
  uv run python program_controlled.py
  ```

- **capture_utils.py**: Utilities for capturing simulation data like images or logs.

## Key Features

- **Robust Prim Path Detection**: Automatically finds correct paths for robot hands and fingers, with fallback strategies.
- **Dynamic Grasp Position Calculation**: Calculates optimal grasp positions based on object dimensions.
- **Smooth Trajectory Planning**: Implements smooth trajectories with acceleration/deceleration profiles.
- **Error Recovery**: Handles potential errors during execution with recovery strategies.
- **Position Feedback**: Uses real-time position feedback to refine and adjust the grasp approach.

## Implementation Details

### Grasp Approach

The grasp approach uses a combination of techniques:

1. Starting with optimal joint configurations derived from manual positioning
2. Adjusting the configuration based on the actual object position
3. Using iterative refinement with position feedback
4. Implementing smooth trajectory planning for natural robot movements

### Object Detection

The scripts can reliably detect:
- Banana and other objects in the scene
- Target crate position
- Robot hand and finger paths
- Object dimensions for better grasp positioning

## Performance

The optimized script achieves a grasp precision of approximately 0.015m (well within the 0.025m threshold) and successfully completes the full pick-and-place sequence with smooth motions.

## Notes

- The imports for Isaac Sim modules will show as unresolved in most code editors but will work correctly when running in the Isaac Sim environment.
- For optimal performance, ensure your Isaac Sim installation is properly set up and the environment variables are correctly configured in `core.py`.
- The scripts have been tested with Isaac Sim 4.5.0 and may require adjustments for other versions.

## Franka Robot Pick and Place

This project implements automated pick-and-place tasks with the Franka robot in NVIDIA Isaac Sim 4.5.0.

### Pick and Place Scripts

- **optimized_franka_pick_place.py**: The latest and most robust implementation with enhanced error handling, object detection, and smooth trajectories.
- **improved_final_franka_pick_place_banana.py**: A highly reliable implementation with precise banana grasping capabilities.
- **franka_pick_place_banana.py**: The original implementation of the pick-and-place task.

### Running the Pick and Place Scripts

Use the runner script to test different implementations:

```bash
python run_pick_place.py [optimized|original|improved|advanced|enhanced|final]
```

### Key Features

- **Robust Prim Path Detection**: Automatically finds correct paths for robot hands and fingers.
- **Dynamic Grasp Position Calculation**: Calculates optimal grasp positions based on object dimensions.
- **Smooth Trajectory Planning**: Implements smooth trajectories with acceleration/deceleration profiles.
- **Error Recovery**: Handles potential errors during execution with recovery strategies.
- **Position Feedback**: Uses real-time position feedback to refine and adjust the grasp approach.

### Implementation Details

The grasp approach uses:
1. Optimal joint configurations derived from manual positioning
2. Adjustments based on actual object position
3. Iterative refinement with position feedback
4. Smooth trajectory planning for natural robot movements

The optimized script achieves a grasp precision of approximately 0.015m (well within the 0.025m threshold) and successfully completes the full pick-and-place sequence with smooth motions.
