"""
Isaac Sim Screenshot and Joint Position Capture Example

This module demonstrates how to capture screenshots and robot joint positions
during Isaac Sim simulation. The implementation is robust and falls back to
joint position capture only if screenshot functionality is unavailable.

Key features:
- Captures viewport screenshots using Isaac Sim's built-in commands
- Records robot joint positions for each action
- Saves data in JSON format with timestamps
- Creates organized output directory structure
- Handles errors gracefully with fallback options
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


def capture_simulation_data(
    world,
    robot_name: str,
    cube_name: str,
    action_number: int,
    output_dir: Path,
    enable_screenshots: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive data capture function for Isaac Sim simulations.

    Args:
        world: Isaac Sim World object
        robot_name: Name of the robot in the scene
        cube_name: Name of the cube object in the scene
        action_number: Current action step number
        output_dir: Directory to save captured data
        enable_screenshots: Whether to attempt screenshot capture

    Returns:
        Dictionary containing captured data
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds

    # Get current observations
    observations = world.get_observations()
    robot_observations = observations[robot_name]
    cube_observations = observations[cube_name]

    # Extract robot state data
    joint_positions = robot_observations["joint_positions"].tolist()

    # Attempt screenshot capture
    screenshot_filename = None
    screenshot_success = False

    if enable_screenshots:
        try:
            # Method 1: Try Isaac Sim's ExportScreenshot command
            screenshot_filename = f"action_{action_number:03d}_{timestamp}.png"
            screenshot_path = output_dir / screenshot_filename

            import omni.kit.commands

            omni.kit.commands.execute(
                "ExportScreenshot",
                file_path=str(screenshot_path),
                camera_viewport=False,
                width=1024,
                height=768,
            )

            # Verify the file was created
            if screenshot_path.exists() and screenshot_path.stat().st_size > 0:
                screenshot_success = True
                print(f"✓ Screenshot captured: {screenshot_filename}")
            else:
                screenshot_filename = None

        except Exception as e:
            print(f"⚠ Screenshot method 1 failed: {e}")
            screenshot_filename = None

        # Method 2: Alternative screenshot approach if first method failed
        if not screenshot_success:
            try:
                import omni.kit.viewport.utility as vp_utils

                viewport_api = vp_utils.get_active_viewport()
                if viewport_api:
                    screenshot_filename = (
                        f"action_{action_number:03d}_{timestamp}_alt.png"
                    )
                    screenshot_path = output_dir / screenshot_filename

                    # This is an alternative approach - exact API may vary by Isaac Sim version
                    viewport_api.export_to_file(str(screenshot_path))

                    if screenshot_path.exists():
                        screenshot_success = True
                        print(
                            f"✓ Alternative screenshot captured: {screenshot_filename}"
                        )
                    else:
                        screenshot_filename = None

            except Exception as e:
                print(f"⚠ Screenshot method 2 failed: {e}")
                screenshot_filename = None

    # Compile comprehensive scene data
    scene_data = {
        "metadata": {
            "timestamp": timestamp,
            "action_number": action_number,
            "isaac_sim_version": "4.5.0",  # Could be detected dynamically
            "screenshot_captured": screenshot_success,
        },
        "robot_state": {
            "name": robot_name,
            "joint_positions": joint_positions,
            "joint_count": len(joint_positions),
        },
        "scene_objects": {
            "cube": {
                "name": cube_name,
                "current_position": cube_observations["position"].tolist(),
                "target_position": cube_observations["target_position"].tolist(),
            }
        },
        "capture_info": {
            "screenshot_filename": screenshot_filename,
            "data_directory": str(output_dir),
        },
    }

    # Save JSON data file
    json_filename = f"action_{action_number:03d}_{timestamp}.json"
    json_path = output_dir / json_filename

    with open(json_path, "w") as f:
        json.dump(scene_data, f, indent=2, sort_keys=True)

    # Print capture summary
    print(f"📊 Action {action_number} data captured:")
    print(f"   • Joint positions: {len(joint_positions)} joints")
    print(f"   • Cube position: {cube_observations['position'].tolist()}")
    print(f"   • Screenshot: {'✓' if screenshot_success else '✗'}")
    print(f"   • Data file: {json_filename}")

    return scene_data


def setup_capture_directory(base_name: str = "isaac_sim_capture") -> Path:
    """
    Create a timestamped directory for capturing simulation data.

    Args:
        base_name: Base name for the capture directory

    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{base_name}_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (output_dir / "screenshots").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)

    # Create a README file explaining the data structure
    readme_content = f"""# Isaac Sim Capture Data - {timestamp}

This directory contains captured data from an Isaac Sim simulation session.

## File Structure:
- action_XXX_YYYYMMDD_HHMMSS_mmm.json: Simulation state data for each action
- action_XXX_YYYYMMDD_HHMMSS_mmm.png: Screenshots (if captured successfully)
- screenshots/: Additional screenshot storage (if used)
- metadata/: Additional metadata files (if used)

## Data Format:
Each JSON file contains:
- metadata: Timestamp, action number, version info
- robot_state: Joint positions and robot information
- scene_objects: Positions of objects in the scene
- capture_info: Information about captured files

## Usage:
This data can be used for:
- Robot behavior analysis
- Training machine learning models
- Debugging simulation scenarios
- Creating datasets for research

Generated by Isaac Sim capture system on {datetime.now().isoformat()}
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)

    return output_dir
