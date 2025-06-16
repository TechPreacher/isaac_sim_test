import time
import warnings
from pathlib import Path
from core import setup_isaac_sim_environment
from robot_movements import RobotMovements

# Suppress numpy compatibility warnings
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

THIS_FOLDER = Path(__file__).parent.resolve()


def run_robot_basic():
    """
    A simplified version that uses a direct approach to access the Franka robot
    at path "/World/franka" using the most basic APIs.
    """
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})
    
    # Import necessary modules after simulation app is initialized
    import numpy as np
    import omni.usd
    
    try:
        # Try importing from isaacsim (newer versions)
        from isaacsim.core.api import World
        print("Using isaacsim.core.api API")
    except ImportError:
        # Fall back to older omni.isaac APIs
        from omni.isaac.core.world import World
        print("Using omni.isaac.core.world API")

    # Open the scene
    omni_usd = omni.usd.get_context()
    omni_usd.open_stage(str(THIS_FOLDER / "scenes" / "franka_blocks_manual.usd"))
    simulation_app.update()
    
    # Create the world
    my_world = World()
    
    # Reset the world - this will initialize all objects in the scene
    my_world.reset()
    
    # Step the simulation a few times to ensure the scene is properly loaded
    for _ in range(20):
        my_world.step(render=True)
    
    # Try to find the robot directly
    robot_found = False
    try:
        # Try to get the robot by name 'franka'
        my_franka = my_world.scene.get_object("franka")
        print("Found robot by name 'franka'")
        robot_found = True
    except Exception as e:
        print(f"Could not find robot by name 'franka': {e}")
        
        # Try alternative approaches if the direct approach failed
        if not robot_found:
            try:
                # Try to get all objects in the scene
                print("Looking for Franka among all objects...")
                object_names = my_world.scene.get_object_names()
                print(f"Available objects: {object_names}")
                
                # Look for anything that might be the Franka robot
                for name in object_names:
                    if "franka" in name.lower():
                        my_franka = my_world.scene.get_object(name)
                        print(f"Found robot by name: {name}")
                        robot_found = True
                        break
            except Exception as e:
                print(f"Error getting object names: {e}")
    
    # If we still couldn't find the robot, exit
    if not robot_found:
        print("ERROR: Could not find the Franka robot in the scene.")
        simulation_app.close()
        return
    
    # Initialize the robot movements controller
    print("Initializing robot movements controller...")
    robot_movements = RobotMovements(my_franka, max_velocity=0.2)
    
    # Run the simulation
    movement_count = 0
    max_movements = 10  # Number of random movements to perform
    movement_interval = 200  # Steps between movements
    
    print("Starting simulation with random joint movements...")
    print(f"Robot has {len(my_franka.dof_names)} joints: {my_franka.dof_names}")
    
    while simulation_app.is_running() and movement_count < max_movements:
        my_world.step(render=True)
        
        if my_world.current_time_step_index % movement_interval == 0 and movement_count < max_movements:
            print(f"\nMovement {movement_count + 1}/{max_movements}")
            random_positions = robot_movements.set_random_joint_positions(scale_factor=0.4)
            print(f"Random joint positions: {random_positions}")
            movement_count += 1
            
            # Wait for a moment to see the effect
            for _ in range(100):
                my_world.step(render=True)
                
    # Move back to home position before ending
    print("\nMoving back to home position...")
    robot_movements.move_to_home_position()
    for _ in range(100):
        my_world.step(render=True)
            
    print("Simulation complete.")
    simulation_app.close()


if __name__ == "__main__":
    # Set up the Isaac Sim environment and run the simulation
    setup_isaac_sim_environment()
    # Use the basic approach for maximum compatibility
    run_robot_basic()
