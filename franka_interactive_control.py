#!/usr/bin/env python3
"""
Franka Robot Interactive Control in Isaac Sim

This script loads the Franka robot from a USD scene in Isaac Sim and allows
interactive control of the robot's joints using keyboard inputs.
1. Load a USD scene with a Franka robot
2. Access the robot directly at the prim path "/World/franka"
3. Control the robot's joints with keyboard input

This is a clean, robust approach that works with Isaac Sim 4.5.0.

Keyboard Controls:
- 1-7: Select joints 1-7 (arm joints)
- 8: Select both finger joints
- Q/A: Increase/decrease the selected joint position
- R: Reset all joints to initial position
- ESC: Exit the application
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

def run_franka_interactive_control():
    """
    Run Isaac Sim simulation with a Franka robot and control its joints with keyboard
    """
    # Import SimulationApp first to initialize the simulation
    from isaacsim import SimulationApp

    # Create the simulation app (use headless=True for running without UI)
    simulation_app = SimulationApp({"headless": False})
    
    # Import necessary modules after SimulationApp is initialized
    import omni.usd
    import omni.timeline
    import omni.kit.commands
    
    # Import UI and input modules
    from omni.kit.window.popup_dialog import PopupDialog
    import carb.input
    
    log("Starting Franka robot interactive control")
    
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
    
    # Try both modern and deprecated APIs
    robot = None
    joint_positions = None
    
    # First try the modern Isaac Sim 4.5.0 API
    try:
        # Modern approach for Isaac Sim 4.5+
        log("Trying modern isaacsim.core.prims.SingleArticulation API...")
        from isaacsim.core.prims import SingleArticulation
        
        # Create articulation from existing prim
        robot = SingleArticulation(prim_path=ROBOT_PRIM_PATH, name="franka")
        
        # Update app to ensure articulation is registered
        simulation_app.update()
        time.sleep(0.2)
        
        # Initialize the robot
        robot.initialize()
        log("Successfully initialized the robot using modern articulation API")
        
        # Get joint information
        dof_names = robot.dof_names
        dof_count = robot.num_dof
        log(f"Robot has {dof_count} degrees of freedom")
        log(f"Joint names: {dof_names}")
        
        # Get current joint positions
        joint_positions = robot.get_joint_positions()
        log(f"Initial joint positions: {joint_positions}")
        
    except Exception as e:
        log(f"Could not initialize robot with modern API: {e}")
        log("Trying with deprecated API as fallback...")
        
        try:
            # Fallback to deprecated API
            log("Trying deprecated omni.isaac.core.articulations.Articulation API...")
            from omni.isaac.core.articulations import Articulation
            
            # Create articulation from existing prim
            robot = Articulation(prim_path=ROBOT_PRIM_PATH, name="franka")
            
            # Update app to ensure articulation is registered
            simulation_app.update()
            time.sleep(0.2)
            
            # Initialize the robot
            robot.initialize()
            log("Successfully initialized the robot using deprecated articulation API")
            
            # Get joint information
            dof_names = robot.dof_names
            dof_count = robot.num_dof
            log(f"Robot has {dof_count} degrees of freedom")
            log(f"Joint names: {dof_names}")
            
            # Get current joint positions
            joint_positions = robot.get_joint_positions()
            log(f"Initial joint positions: {joint_positions}")
            
        except Exception as e2:
            log(f"Could not initialize robot articulation with either API: {e2}")
            import traceback
            traceback.print_exc()
    
    # If we successfully initialized the robot and got joint positions
    if robot is not None and joint_positions is not None:
        # Set up keyboard input handling
        input_interface = carb.input.acquire_input_interface()
        keyboard = input_interface.create_keyboard()
        
        # Store initial joint positions for reset function
        initial_positions = joint_positions.copy()
        current_positions = joint_positions.copy()
        
        # Track which joint is currently selected (0-based index)
        selected_joint = 0
        
        # Joint movement step size
        step_size = 0.05
        
        # Display initial instructions
        log("Keyboard controls ready:")
        log("  1-7: Select arm joints 1-7")
        log("  8: Select both finger joints")
        log("  Q/A: Increase/decrease selected joint position")
        log("  R: Reset all joints to initial position")
        log("  ESC: Exit the application")
        log(f"Currently selected joint: {dof_names[selected_joint]}")
        
        # Create a popup with instructions
        popup = PopupDialog(
            title="Franka Robot Keyboard Control",
            message=("Keyboard Controls:\n"
                    "1-7: Select joints 1-7 (arm joints)\n"
                    "8: Select both finger joints\n"
                    "Q/A: Increase/decrease the selected joint position\n"
                    "R: Reset all joints to initial position\n"
                    "ESC: Exit the application"),
            ok_label="OK"
        )
        
        # Main control loop
        running = True
        while running and simulation_app.is_running():
            # Process input
            
            # Joint selection with number keys 1-8
            for i in range(1, 9):
                if keyboard.was_pressed(getattr(carb.input.KeyboardInput, f"KEY_{i}")):
                    if i <= 7:
                        selected_joint = i - 1
                        log(f"Selected joint: {dof_names[selected_joint]}")
                    else:
                        # Key 8 selects both finger joints (7 and 8 in 0-based indexing)
                        selected_joint = 7  # First finger joint
                        log(f"Selected both finger joints: {dof_names[7]} and {dof_names[8]}")
            
            # Joint control with Q/A keys
            if keyboard.was_pressed(carb.input.KeyboardInput.KEY_q):
                if selected_joint < 7:
                    # Single joint control for arm joints
                    current_positions[selected_joint] += step_size
                    log(f"Increasing {dof_names[selected_joint]} to {current_positions[selected_joint]:.4f}")
                else:
                    # Control both finger joints simultaneously
                    current_positions[7] += step_size
                    current_positions[8] += step_size
                    log(f"Increasing both finger joints to {current_positions[7]:.4f}")
                
                # Apply the new position
                robot.set_joint_positions(current_positions)
            
            if keyboard.was_pressed(carb.input.KeyboardInput.KEY_a):
                if selected_joint < 7:
                    # Single joint control for arm joints
                    current_positions[selected_joint] -= step_size
                    log(f"Decreasing {dof_names[selected_joint]} to {current_positions[selected_joint]:.4f}")
                else:
                    # Control both finger joints simultaneously
                    current_positions[7] -= step_size
                    current_positions[8] -= step_size
                    log(f"Decreasing both finger joints to {current_positions[7]:.4f}")
                
                # Apply the new position
                robot.set_joint_positions(current_positions)
            
            # Reset to initial position with R key
            if keyboard.was_pressed(carb.input.KeyboardInput.KEY_r):
                current_positions = initial_positions.copy()
                robot.set_joint_positions(current_positions)
                log("Reset to initial position")
            
            # Exit with ESC key
            if keyboard.was_pressed(carb.input.KeyboardInput.KEY_ESCAPE):
                log("Exiting interactive control")
                running = False
            
            # Update simulation
            simulation_app.update()
            time.sleep(0.01)  # Control simulation speed
    else:
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
    run_franka_interactive_control()
