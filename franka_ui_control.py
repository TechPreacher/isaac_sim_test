#!/usr/bin/env python3
"""
Franka Robot Interactive Control in Isaac Sim

This script loads the Franka robot from a USD scene in Isaac Sim and allows
interactive control of the robot's joints using a UI window with buttons.
1. Load a USD scene with a Franka robot
2. Access the robot directly at the prim path "/World/franka"
3. Control the robot's joints with UI buttons and keyboard shortcuts

This is a clean, robust approach that works with Isaac Sim 4.5.0.

Controls:
- Select joints 1-7 (arm joints) or gripper using buttons
- Increase/decrease joint position with buttons or Q/A keys
- Reset all joints to initial position
- Adjust movement step size
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
    Run Isaac Sim simulation with a Franka robot and control its joints with UI
    """
    # Import SimulationApp first to initialize the simulation
    from isaacsim import SimulationApp

    # Create the simulation app (use headless=True for running without UI)
    simulation_app = SimulationApp({"headless": False})
    
    # Import necessary modules after SimulationApp is initialized
    import omni.usd
    import omni.timeline
    import omni.ui as ui
    
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
        # Store initial joint positions for reset function
        initial_positions = joint_positions.copy()
        current_positions = joint_positions.copy()
        
        # Track which joint is currently selected (0-based index)
        selected_joint = 0
        
        # Joint movement step size
        step_size = 0.05
        
        # Create UI elements for controlling the robot
        window = ui.Window("Franka Robot Control", width=400, height=600)
        with window.frame:
            with ui.VStack(spacing=5, height=0):
                ui.Label("Franka Robot Joint Control")
                
                # Display selected joint
                joint_label = ui.Label(f"Selected Joint: {dof_names[selected_joint]}")
                
                # Joint selection buttons
                with ui.HStack(spacing=5):
                    ui.Label("Select Joint:")
                    
                    for i in range(1, 8):
                        def make_select_callback(joint_idx):
                            def callback():
                                nonlocal selected_joint
                                selected_joint = joint_idx
                                joint_label.text = f"Selected Joint: {dof_names[selected_joint]}"
                                log(f"Selected joint: {dof_names[selected_joint]}")
                            return callback
                        
                        ui.Button(str(i), clicked_fn=make_select_callback(i-1))
                    
                    # Button for both finger joints
                    def select_fingers():
                        nonlocal selected_joint
                        selected_joint = 7  # First finger joint
                        joint_label.text = f"Selected: Both Finger Joints"
                        log(f"Selected both finger joints: {dof_names[7]} and {dof_names[8]}")
                    
                    ui.Button("Grip", clicked_fn=select_fingers)
                
                # Joint control buttons
                with ui.HStack(spacing=10):
                    def increase_joint():
                        nonlocal current_positions
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
                    
                    def decrease_joint():
                        nonlocal current_positions
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
                    
                    ui.Button("Increase (Q)", clicked_fn=increase_joint)
                    ui.Button("Decrease (A)", clicked_fn=decrease_joint)
                
                # Reset button
                def reset_joints():
                    nonlocal current_positions
                    current_positions = initial_positions.copy()
                    robot.set_joint_positions(current_positions)
                    log("Reset to initial position")
                
                ui.Button("Reset All Joints (R)", clicked_fn=reset_joints)
                
                # Step size control
                with ui.HStack(spacing=5):
                    ui.Label("Step Size:")
                    
                    def set_step_size(size):
                        nonlocal step_size
                        step_size = size
                        log(f"Step size set to {step_size:.3f}")
                    
                    ui.Button("Small (0.01)", clicked_fn=lambda: set_step_size(0.01))
                    ui.Button("Medium (0.05)", clicked_fn=lambda: set_step_size(0.05))
                    ui.Button("Large (0.1)", clicked_fn=lambda: set_step_size(0.1))
                
                # Instructions
                with ui.VStack(spacing=5):
                    ui.Label("Instructions:")
                    ui.Label("1. Select a joint using the numbered buttons")
                    ui.Label("2. Use Increase/Decrease buttons to move the joint")
                    ui.Label("3. Click Reset to return to initial position")
                    ui.Label("4. Adjust step size as needed")

                # Status display
                status_label = ui.Label("Ready")
                  # Add keyboard shortcuts (register with UI system)
                def on_keyboard_event(event):
                    nonlocal selected_joint
                    # Number keys 1-8
                    if event.type == ui.KeyboardEventType.KEY_PRESSED:
                        if event.code >= ui.KeyCode.ONE and event.code <= ui.KeyCode.SEVEN:
                            # Convert to 0-based index (1->0, 2->1, etc.)
                            joint_idx = event.code - ui.KeyCode.ONE
                            selected_joint = joint_idx
                            joint_label.text = f"Selected Joint: {dof_names[selected_joint]}"
                            log(f"Selected joint: {dof_names[selected_joint]}")
                            return True
                        
                        elif event.code == ui.KeyCode.EIGHT:
                            # Select finger joints
                            selected_joint = 7
                            joint_label.text = f"Selected: Both Finger Joints"
                            log(f"Selected both finger joints: {dof_names[7]} and {dof_names[8]}")
                            return True
                        
                        elif event.code == ui.KeyCode.Q:
                            increase_joint()
                            return True
                            
                        elif event.code == ui.KeyCode.A:
                            decrease_joint()
                            return True
                            
                        elif event.code == ui.KeyCode.R:
                            reset_joints()
                            return True
                            
                    return False
                
                # Register the keyboard event handler
                window.subscribe_to_keyboard_events(on_keyboard_event)
        
        # Main control loop
        log("UI control window created. You can now control the robot.")
        log("Keyboard shortcuts: 1-7 to select joints, 8 for gripper, Q/A to move, R to reset")
        
        try:
            while simulation_app.is_running():
                # Update simulation
                simulation_app.update()
                time.sleep(0.01)  # Control simulation speed
        except Exception as e:
            log(f"Error during simulation: {e}")
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
    
    log("Simulation complete")
    simulation_app.close()

if __name__ == "__main__":
    # Set up the environment and run the simulation
    setup_isaac_sim_environment()
    run_franka_interactive_control()
