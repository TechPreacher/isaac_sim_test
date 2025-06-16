#!/usr/bin/env python3

import time
import omni
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.simulation_context import SimulationContext
import omni.kit.commands

# Path to the USD scene file
USD_PATH = "scenes/franka_blocks_manual.usd"
ROBOT_PRIM_PATH = "/World/franka"

def main():
    # Initialize the simulation
    simulation_context = SimulationContext(physics_dt=1.0/60.0, rendering_dt=1.0/60.0, stage_units_in_meters=1.0)
    
    # Open the stage with the Franka robot
    print(f"Opening USD stage: {USD_PATH}")
    open_stage(USD_PATH)
    
    # Get the robot prim
    robot_prim = omni.usd.get_context().get_stage().GetPrimAtPath(ROBOT_PRIM_PATH)
    if not robot_prim:
        print(f"ERROR: Could not find robot at path '{ROBOT_PRIM_PATH}'")
        return

    print(f"Found robot prim: {ROBOT_PRIM_PATH}")
    print(f"Robot prim type: {robot_prim.GetTypeName()}")
    
    # Play the simulation
    simulation_context.play()
    
    try:
        # Try different methods to get the articulation controller
        print("Starting simulation and trying to access the articulation controller...")
        
        # Method 1: Using the Physics Scene API
        try:
            from omni.physics.tensors import simulate
            from omni.physics import get_physics_context
            
            physics_context = get_physics_context()
            physics_scene = physics_context.get_scene()
            
            print("Using physics scene API to find articulations...")
            # Step the simulation a few times to make sure physics is initialized
            for i in range(10):
                simulation_context.step(render=True)
                time.sleep(0.1)
            
            # Try to get the articulation
            try:
                # For newer Isaac Sim versions
                import omni.physics.tensors
                articulation_view = omni.physics.tensors.get_articulation_view()
                articulation_count = articulation_view.count
                print(f"Found {articulation_count} articulations in the scene")
                
                if articulation_count > 0:
                    print("Articulation properties:")
                    # Print joint information if available
                    try:
                        joints_view = omni.physics.tensors.get_joint_view()
                        print(f"Found {joints_view.count} joints")
                    except Exception as e:
                        print(f"Error getting joints: {e}")
            except Exception as e:
                print(f"Error using tensor API: {e}")
                
            # Alternate method: Check for articulation via prim properties
            print("\nChecking robot prim properties for articulation data:")
            for prop_name in robot_prim.GetPropertyNames():
                if "articulation" in prop_name.lower():
                    property_value = robot_prim.GetProperty(prop_name).Get()
                    print(f"  - {prop_name}: {property_value}")
        
        except Exception as e:
            print(f"Error accessing physics scene: {e}")
        
        # Method 2: Try using Articulation API directly
        try:
            from omni.isaac.core.articulations import Articulation
            
            print("\nTrying to access the robot using Articulation API...")
            # Step the simulation a few more times
            for i in range(5):
                simulation_context.step(render=True)
                time.sleep(0.1)
                
            # Try to create articulation from existing prim
            try:
                robot = Articulation(prim_path=ROBOT_PRIM_PATH, name="franka")
                simulation_context.step(render=True)
                time.sleep(0.1)
                
                # Initialize the robot
                robot.initialize()
                print("Successfully initialized the robot articulation!")
                
                # Get joint information
                dof_names = robot.dof_names
                dof_count = robot.num_dof
                print(f"Robot has {dof_count} degrees of freedom")
                print(f"Joint names: {dof_names}")
                
                # Get current joint positions
                joint_positions = robot.get_joint_positions()
                print(f"Current joint positions: {joint_positions}")
                
                # Run simulation loop with robot control
                print("Running simulation for 5 seconds...")
                start_time = time.time()
                while time.time() - start_time < 5.0:
                    # Simple joint position command - just move slightly from current position
                    target_positions = joint_positions.copy()
                    if len(target_positions) > 0:
                        target_positions[0] += 0.1  # Move the first joint a bit
                    
                    # Apply joint commands
                    robot.set_joint_positions(target_positions)
                    
                    # Step simulation
                    simulation_context.step(render=True)
                    time.sleep(0.05)
                    
                    # Update joint positions for next iteration
                    joint_positions = robot.get_joint_positions()
                
            except Exception as e:
                print(f"Error initializing robot articulation: {e}")
                print("Trying alternative methods...")
        
        except Exception as e:
            print(f"Error importing Articulation API: {e}")
        
        # Let the simulation run for a few seconds so we can see the robot
        print("\nRunning simulation for 10 seconds to visualize the robot...")
        start_time = time.time()
        while time.time() - start_time < 10.0:
            simulation_context.step(render=True)
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        # Stop the simulation
        print("Stopping simulation...")
        simulation_context.stop()

if __name__ == "__main__":
    main()
