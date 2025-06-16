#!/usr/bin/env python3
"""
Franka Robot Control in Isaac Sim

This script loads the Franka robot from a USD scene in Isaac Sim and controls it
directly without using the PickPlace task. It demonstrates how to:
1. Load a USD scene with a Franka robot
2. Access the robot directly at the prim path "/World/franka"
3. Control the robot's joints with a simple motion pattern

This is a clean, robust approach that should work across different Isaac Sim versions.
"""

import time
import warnings
import numpy as np
from pathlib import Path
from core import setup_isaac_sim_environment

# Suppress numpy compatibility warnings
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

THIS_FOLDER = Path(__file__).parent.resolve()
USD_SCENE_PATH = str(THIS_FOLDER / "scenes" / "franka_blocks_manual.usd")
ROBOT_PRIM_PATH = "/World/franka"

def log(message):
    """Helper function for logging with timestamps"""
    print(f"[{time.time():.2f}s] {message}")

def run_franka_control():
    """
    Run Isaac Sim simulation with a Franka robot and control its joints
    """
    # Import SimulationApp first to initialize the simulation
    from isaacsim import SimulationApp

    # Create the simulation app (use headless=True for running without UI)
    simulation_app = SimulationApp({"headless": False})
    
    # Import necessary modules after SimulationApp is initialized
    import omni.usd
    import omni.timeline
    
    log("Starting Franka robot control demonstration")
    
    # Open the USD stage with the scene
    log(f"Opening scene: {USD_SCENE_PATH}")
    usd_context = omni.usd.get_context()
    usd_context.open_stage(USD_SCENE_PATH)
    simulation_app.update()
    
    # Get the stage after opening it
    stage = usd_context.get_stage()
    
    # Check if the robot exists in the stage
    robot_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    if not robot_prim:
        log(f"ERROR: Robot not found at {ROBOT_PRIM_PATH}")
        simulation_app.close()
        return
    
    log(f"Found robot prim at {ROBOT_PRIM_PATH} with type: {robot_prim.GetTypeName()}")
    
    # Get timeline interface for controlling playback
    timeline = omni.timeline.get_timeline_interface()
    
    # Start the simulation
    timeline.play()
    simulation_app.update()
    
    # Wait for physics to initialize
    log("Initializing physics simulation...")
    for i in range(10):
        simulation_app.update()
        time.sleep(0.1)
    
    # Access the robot as an articulation
    log("Accessing the robot as an articulation...")
    
    # Import the articulation module - use the proper import based on Isaac Sim version
    try:
        # Modern approach (4.0+)
        from omni.isaac.core.articulations import Articulation
        
        # Create articulation from existing prim
        robot = Articulation(prim_path=ROBOT_PRIM_PATH, name="franka")
        
        # Update app to ensure articulation is registered
        simulation_app.update()
        time.sleep(0.2)
        
        # Initialize the robot
        robot.initialize()
        log("Successfully initialized the robot articulation")
        
        # Get joint information
        dof_names = robot.dof_names
        dof_count = robot.num_dof
        log(f"Robot has {dof_count} degrees of freedom")
        log(f"Joint names: {dof_names}")
        
        # Get current joint positions
        joint_positions = robot.get_joint_positions()
        log(f"Initial joint positions: {joint_positions}")
        
        # Define different motion patterns
        patterns = {
            "wave": lambda t, jp: jp.copy() + np.array([0.5 * np.sin(t), 0, 0, 0, 0, 0, 0, 0, 0]),
            "circular": lambda t, jp: jp.copy() + np.array([0.3 * np.sin(t), 0.3 * np.cos(t), 0, 0, 0, 0, 0, 0, 0]),
            "alternating": lambda t, jp: jp.copy() + np.array([0.3 * np.sin(t), 0, 0.3 * np.cos(t), 0, 0, 0, 0, 0, 0]),
            "grip": lambda t, jp: jp.copy() + np.array([0, 0, 0, 0, 0, 0, 0, 0.02 * np.sin(t), 0.02 * np.sin(t)]),
            "reset": lambda t, jp: np.zeros(jp.shape)  # Reset to zero position
        }
        
        # Run the simulation with joint control
        total_steps = 1500
        step_count = 0
        
        log("Running simulation with joint control...")
        log("Demonstrating different motion patterns...")
        
        # Track which pattern we're using
        current_pattern_name = "wave"
        current_pattern = patterns[current_pattern_name]
        pattern_start_step = 0
        steps_per_pattern = 300
        
        # Store original joint positions for reference
        original_positions = joint_positions.copy()
        
        try:
            while simulation_app.is_running() and step_count < total_steps:
                # Check if we need to switch patterns
                if step_count > 0 and (step_count - pattern_start_step) % steps_per_pattern == 0:
                    # Cycle through patterns
                    if current_pattern_name == "wave":
                        current_pattern_name = "circular"
                    elif current_pattern_name == "circular":
                        current_pattern_name = "alternating"
                    elif current_pattern_name == "alternating":
                        current_pattern_name = "grip"
                    elif current_pattern_name == "grip":
                        current_pattern_name = "reset"
                    elif current_pattern_name == "reset":
                        current_pattern_name = "wave"
                    
                    current_pattern = patterns[current_pattern_name]
                    pattern_start_step = step_count
                    log(f"Switching to '{current_pattern_name}' pattern")
                
                # Calculate time parameter relative to pattern start
                t = (step_count - pattern_start_step) / 50.0
                
                # Get new positions based on current pattern
                if current_pattern_name == "reset":
                    # Special case - reset to original position gradually
                    progress = min(1.0, (step_count - pattern_start_step) / 100.0)
                    new_positions = original_positions * (1 - progress)
                else:
                    new_positions = current_pattern(t, original_positions)
                
                # Apply joint positions
                robot.set_joint_positions(new_positions)
                
                # Update simulation
                simulation_app.update()
                
                # Print status periodically
                if step_count % 100 == 0:
                    log(f"Step {step_count}/{total_steps}, Pattern: {current_pattern_name}")
                
                step_count += 1
                time.sleep(0.01)  # Control simulation speed
                
        except Exception as e:
            log(f"Error during joint control: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        log(f"Could not initialize robot articulation: {e}")
        import traceback
        traceback.print_exc()
        
        # Run without control if we couldn't access the articulation
        log("Running simulation without joint control (visualization only)...")
        total_steps = 500
        for step_count in range(total_steps):
            simulation_app.update()
            if step_count % 100 == 0:
                log(f"Step {step_count}/{total_steps}")
            time.sleep(0.01)
    
    log("Simulation complete")
    simulation_app.close()

if __name__ == "__main__":
    # Set up the environment and run the simulation
    setup_isaac_sim_environment()
    run_franka_control()
