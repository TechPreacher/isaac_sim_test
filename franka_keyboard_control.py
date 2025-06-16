#!/usr/bin/env python3
"""
Franka Robot Keyboard Control in Isaac Sim

This script loads the Franka robot from a USD scene in Isaac Sim and allows
interactive control of the robot's joints using keyboard inputs.
1. Load a USD scene with a Franka robot
2. Access the robot directly at the prim path "/World/franka"
3. Control the robot's joints with keyboard inputs using `carb.input` directly

This is a clean, robust approach that works with Isaac Sim 4.5.0.

Controls:
- Number keys 1-7: Select arm joints
- Key 8: Select gripper
- Key Up/Down: Increase/decrease joint position
- Key R: Reset all joints to initial position
- Key PageUp/PageDown: Adjust movement step size
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

def run_franka_keyboard_control():
    """
    Run Isaac Sim simulation with a Franka robot and control its joints with keyboard
    """
    # Import SimulationApp first to initialize the simulation
    from isaacsim import SimulationApp

    # Create the simulation app (use headless=False to enable UI)
    simulation_app = SimulationApp({"headless": False})
    
    # Import necessary modules after SimulationApp is initialized
    import omni.usd
    import omni.timeline
    
    try:
        # Try to import carb.input for keyboard handling
        import carb.input
        input_interface = carb.input.acquire_input_interface()
        keyboard = input_interface.create_keyboard()
        has_keyboard_input = True
        log("Successfully initialized keyboard input")
    except Exception as e:
        log(f"Could not initialize keyboard input: {e}")
        has_keyboard_input = False
    
    log("Starting Franka robot keyboard control")
    
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
    dof_names = None
    
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
    if robot is not None and joint_positions is not None and dof_names is not None:
        # Store initial joint positions for reset function
        initial_positions = joint_positions.copy()
        current_positions = joint_positions.copy()
        
        # Control state
        selected_joint = 0
        step_size = 0.05
        
        # Key states for handling key press/release properly
        key_states = {
            "up": False,      # Increase joint position
            "down": False,    # Decrease joint position
            "r": False,       # Reset joints
            "pageup": False,  # Increase step size
            "pagedown": False # Decrease step size
        }
        
        # Define key codes based on carb.input (if available)
        if has_keyboard_input:
            try:
                # Import KeyboardInput enumeration for key codes
                from carb.input import KeyboardInput
                
                # Key mappings for joint selection
                joint_keys = [
                    KeyboardInput.KEY_1,
                    KeyboardInput.KEY_2,
                    KeyboardInput.KEY_3,
                    KeyboardInput.KEY_4,
                    KeyboardInput.KEY_5,
                    KeyboardInput.KEY_6,
                    KeyboardInput.KEY_7,
                    KeyboardInput.KEY_8,  # For gripper
                ]
                
                # Key mappings for control
                control_keys = {
                    "up": KeyboardInput.KEY_UP,
                    "down": KeyboardInput.KEY_DOWN,
                    "r": KeyboardInput.KEY_R,
                    "pageup": KeyboardInput.KEY_PAGE_UP,
                    "pagedown": KeyboardInput.KEY_PAGE_DOWN
                }
            except Exception as e:
                log(f"Error setting up key codes: {e}")
                has_keyboard_input = False
        
        # Print instructions
        log("\nKeyboard Control Instructions:")
        log("----------------------------")
        log("Number keys 1-7: Select arm joints")
        log("Key 8: Select gripper")
        log("Arrow Up/Down: Increase/decrease joint position")
        log("Key R: Reset all joints to initial position")
        log("PageUp/PageDown: Adjust movement step size")
        log("----------------------------")
        log(f"Currently selected joint: {dof_names[selected_joint]}")
        log(f"Current step size: {step_size}")
        
        # Function to update the joint position based on key input
        def update_joint_position(increment):
            if selected_joint < 7:
                # Single joint control for arm joints
                current_positions[selected_joint] += increment
                log(f"Joint {selected_joint+1} ({dof_names[selected_joint]}) position: {current_positions[selected_joint]:.4f}")
            else:
                # Control both finger joints simultaneously
                current_positions[7] += increment
                current_positions[8] += increment
                log(f"Gripper position: {current_positions[7]:.4f}")
            
            # Apply the new position
            robot.set_joint_positions(current_positions)
        
        # Function to reset all joints
        def reset_all_joints():
            nonlocal current_positions
            current_positions = initial_positions.copy()
            robot.set_joint_positions(current_positions)
            log("Reset to initial position")
        
        # Function to adjust step size
        def adjust_step_size(factor):
            nonlocal step_size
            step_size = max(0.001, min(0.2, step_size * factor))
            log(f"Step size adjusted to: {step_size:.4f}")
        
        # Main control loop
        last_status_time = time.time()
        status_interval = 2.0  # How often to show the status message
        
        try:
            while simulation_app.is_running():
                # Process keyboard input if available
                if has_keyboard_input:
                    # Check for joint selection keys (1-8)
                    for i, key in enumerate(joint_keys):
                        if keyboard.pressed(key) and not keyboard.pressed(key, previous=True):
                            if i < 7:
                                selected_joint = i
                                log(f"Selected joint {i+1}: {dof_names[i]}")
                            else:
                                selected_joint = 7  # Gripper
                                log(f"Selected gripper (joints {dof_names[7]} and {dof_names[8]})")
                    
                    # Check for control keys
                    # Up arrow - increase joint position
                    if keyboard.pressed(control_keys["up"]):
                        if not key_states["up"]:
                            update_joint_position(step_size)
                            key_states["up"] = True
                    else:
                        key_states["up"] = False
                    
                    # Down arrow - decrease joint position
                    if keyboard.pressed(control_keys["down"]):
                        if not key_states["down"]:
                            update_joint_position(-step_size)
                            key_states["down"] = True
                    else:
                        key_states["down"] = False
                    
                    # R key - reset joints
                    if keyboard.pressed(control_keys["r"]):
                        if not key_states["r"]:
                            reset_all_joints()
                            key_states["r"] = True
                    else:
                        key_states["r"] = False
                    
                    # PageUp - increase step size
                    if keyboard.pressed(control_keys["pageup"]):
                        if not key_states["pageup"]:
                            adjust_step_size(1.5)
                            key_states["pageup"] = True
                    else:
                        key_states["pageup"] = False
                    
                    # PageDown - decrease step size
                    if keyboard.pressed(control_keys["pagedown"]):
                        if not key_states["pagedown"]:
                            adjust_step_size(0.67)
                            key_states["pagedown"] = True
                    else:
                        key_states["pagedown"] = False
                
                # Show status message periodically
                current_time = time.time()
                if current_time - last_status_time > status_interval:
                    if has_keyboard_input:
                        log(f"Selected: {'Gripper' if selected_joint == 7 else f'Joint {selected_joint+1}'}, " + 
                            f"Step size: {step_size:.4f}")
                    else:
                        log("Keyboard input not available. Running in visualization mode only.")
                    last_status_time = current_time
                
                # Update simulation
                simulation_app.update()
                time.sleep(0.01)  # Control simulation speed
                
        except Exception as e:
            log(f"Error during joint control: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Run without control if we couldn't access the articulation
        log("Running simulation without joint control (visualization only)...")
        total_steps = 500
        for step_count in range(total_steps):
            simulation_app.update()
            if step_count % 100 == 0:
                log(f"Step {step_count}/{total_steps}")
            time.sleep(0.01)
    
    # Clean up resources
    if has_keyboard_input:
        try:
            input_interface.destroy_keyboard(keyboard)
        except:
            pass
    
    log("Simulation complete")
    simulation_app.close()

if __name__ == "__main__":
    # Set up the environment and run the simulation
    setup_isaac_sim_environment()
    run_franka_keyboard_control()
