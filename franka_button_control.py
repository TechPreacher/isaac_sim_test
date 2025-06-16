#!/usr/bin/env python3
"""
Franka Robot Interactive Control in Isaac Sim

This script loads the Franka robot from a USD scene in Isaac Sim and allows
interactive control of the robot's joints using a UI window with buttons.
1. Load a USD scene with a Franka robot
2. Access the robot directly at the prim path "/World/franka"
3. Control the robot's joints with UI buttons

This is a clean, robust approach that works with Isaac Sim 4.5.0.

Controls:
- Select joints 1-7 (arm joints) or gripper using buttons
- Increase/decrease joint position with buttons
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

    # Create the simulation app (use headless=False to enable UI)
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
        
        # Create reference variables that can be updated inside UI callbacks
        class State:
            selected_joint = 0
            step_size = 0.05
            status_text = "Ready"
        
        # Create UI elements for controlling the robot
        window = ui.Window("Franka Robot Control", width=400, height=700)
        with window.frame:
            with ui.VStack(spacing=5, height=0):
                # Title
                ui.Label("Franka Robot Joint Control", height=30, style={"font_size": 20})
                ui.Spacer(height=10)
                
                # Joint positions display
                positions_frame = ui.Frame(height=0)
                with positions_frame:
                    with ui.VStack(spacing=2):
                        position_labels = []
                        for i in range(len(dof_names)):
                            label = ui.Label(f"{dof_names[i]}: {current_positions[i]:.4f}")
                            position_labels.append(label)
                
                ui.Spacer(height=10)
                
                # Display selected joint with a large, bold font
                joint_label = ui.Label(f"Selected Joint: {dof_names[State.selected_joint]}", 
                                       height=30, 
                                       style={"font_size": 18})
                
                ui.Spacer(height=5)
                
                # Joint selection buttons in a grid layout
                with ui.VStack(spacing=5):
                    ui.Label("Select Joint:")
                    
                    # First row: Joints 1-4
                    with ui.HStack(spacing=5):
                        for i in range(1, 5):
                            def make_select_callback(joint_idx):
                                def callback():
                                    State.selected_joint = joint_idx
                                    joint_label.text = f"Selected Joint: {dof_names[State.selected_joint]}"
                                    log(f"Selected joint: {dof_names[State.selected_joint]}")
                                return callback
                            
                            ui.Button(f"Joint {i}", width=80, height=30, clicked_fn=make_select_callback(i-1))
                    
                    # Second row: Joints 5-7 and Gripper
                    with ui.HStack(spacing=5):
                        for i in range(5, 8):
                            ui.Button(f"Joint {i}", width=80, height=30, 
                                     clicked_fn=make_select_callback(i-1))
                        
                        # Button for both finger joints
                        def select_fingers():
                            State.selected_joint = 7  # First finger joint
                            joint_label.text = f"Selected: Both Gripper Fingers"
                            log(f"Selected both finger joints: {dof_names[7]} and {dof_names[8]}")
                        
                        ui.Button("Gripper", width=80, height=30, clicked_fn=select_fingers)
                
                ui.Spacer(height=10)
                
                # Joint control buttons (larger, more user-friendly)
                with ui.HStack(spacing=10):
                    def increase_joint():
                        if State.selected_joint < 7:
                            # Single joint control for arm joints
                            current_positions[State.selected_joint] += State.step_size
                            log(f"Increasing {dof_names[State.selected_joint]} to {current_positions[State.selected_joint]:.4f}")
                        else:
                            # Control both finger joints simultaneously
                            current_positions[7] += State.step_size
                            current_positions[8] += State.step_size
                            log(f"Opening gripper to {current_positions[7]:.4f}")
                        
                        # Apply the new position
                        robot.set_joint_positions(current_positions)
                        
                        # Update position display
                        for i in range(len(dof_names)):
                            position_labels[i].text = f"{dof_names[i]}: {current_positions[i]:.4f}"
                    
                    def decrease_joint():
                        if State.selected_joint < 7:
                            # Single joint control for arm joints
                            current_positions[State.selected_joint] -= State.step_size
                            log(f"Decreasing {dof_names[State.selected_joint]} to {current_positions[State.selected_joint]:.4f}")
                        else:
                            # Control both finger joints simultaneously
                            current_positions[7] -= State.step_size
                            current_positions[8] -= State.step_size
                            log(f"Closing gripper to {current_positions[7]:.4f}")
                        
                        # Apply the new position
                        robot.set_joint_positions(current_positions)
                        
                        # Update position display
                        for i in range(len(dof_names)):
                            position_labels[i].text = f"{dof_names[i]}: {current_positions[i]:.4f}"
                    
                    ui.Button("+ Increase", width=150, height=40, clicked_fn=increase_joint)
                    ui.Button("- Decrease", width=150, height=40, clicked_fn=decrease_joint)
                
                ui.Spacer(height=10)
                
                # Reset button
                def reset_joints():
                    nonlocal current_positions
                    current_positions = initial_positions.copy()
                    robot.set_joint_positions(current_positions)
                    log("Reset to initial position")
                    
                    # Update position display
                    for i in range(len(dof_names)):
                        position_labels[i].text = f"{dof_names[i]}: {current_positions[i]:.4f}"
                
                ui.Button("Reset All Joints", width=300, height=40, clicked_fn=reset_joints)
                
                ui.Spacer(height=10)
                
                # Step size control
                with ui.VStack(spacing=5):
                    ui.Label("Movement Step Size:")
                    
                    with ui.HStack(spacing=5):
                        def set_step_size(size):
                            State.step_size = size
                            log(f"Step size set to {State.step_size:.3f}")
                        
                        ui.Button("Small (0.01)", width=95, height=30, clicked_fn=lambda: set_step_size(0.01))
                        ui.Button("Medium (0.05)", width=95, height=30, clicked_fn=lambda: set_step_size(0.05))
                        ui.Button("Large (0.1)", width=95, height=30, clicked_fn=lambda: set_step_size(0.1))
                
                ui.Spacer(height=10)
                
                # Instructions
                with ui.Frame(style={"border_width": 1, "border_color": 0xFF999999}):
                    with ui.VStack(spacing=5, style={"margin": 10}):
                        ui.Label("Instructions:", style={"font_size": 16})
                        ui.Label("1. Select a joint using the buttons")
                        ui.Label("2. Use Increase/Decrease buttons to move the joint")
                        ui.Label("3. Click Reset to return to initial position")
                        ui.Label("4. Adjust step size as needed")
                
                ui.Spacer(height=10)
                
                # Status display
                status_label = ui.Label(State.status_text, height=30, style={"font_size": 16})
        
        # Main control loop
        log("UI control window created. You can now control the robot.")
        
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
