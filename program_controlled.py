import time
import warnings
from pathlib import Path
from core import setup_isaac_sim_environment
from robot_movements import RobotMovements

# Suppress numpy compatibility warnings
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

THIS_FOLDER = Path(__file__).parent.resolve()


def run_random_joint_movement():
    """
    Run Isaac Sim simulation with a Franka robot and demonstrate
    random joint movements without using the PickPlace task.
    """
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})
    
    # Basic imports
    import numpy as np
    import importlib
    import sys
    
    # Dynamically import required modules
    def try_import(module_name):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            return None
    
    # Try importing omni.usd
    omni_usd = try_import("omni.usd")
    if not omni_usd:
        print("ERROR: Could not import omni.usd. Make sure Isaac Sim is properly set up.")
        simulation_app.close()
        return
    
    # Open the scene
    omni_usd.get_context().open_stage(str(THIS_FOLDER / "scenes" / "franka_blocks_manual.usd"))
    simulation_app.update()
    
    # Try importing world modules from different APIs
    isaacsim_world = try_import("isaacsim.core.api")
    omni_isaac_world = try_import("omni.isaac.core.world")
    
    # Determine which API to use
    if isaacsim_world:
        print("Using isaacsim API")
        World = isaacsim_world.World
    elif omni_isaac_world:
        print("Using omni.isaac API")
        World = omni_isaac_world.World
    else:
        print("ERROR: Could not import World class from either API")
        simulation_app.close()
        return
    
    # Set up the world
    my_world = World(stage_units_in_meters=1.0)
    my_world.reset()
    
    # Get the stage
    stage = omni_usd.get_context().get_stage()
      # Use the known robot path directly
    robot_prim_path = "/World/franka"
    print(f"Using known Franka robot path: {robot_prim_path}")
    
    # Check if the robot exists at the specified path
    if not stage.GetPrimAtPath(robot_prim_path):
        print(f"ERROR: No robot found at {robot_prim_path}")
        print("Make sure the Franka robot exists in the USD file and try again.")
        simulation_app.close()
        return
    
    # Try different methods to load the robot
    my_franka = None
    robot_found = False
    
    # Method 1: Try using isaacsim.robots.robot.Robot
    try:
        robot_module = try_import("isaacsim.robots.robot")
        if robot_module:
            my_franka = robot_module.Robot(prim_path=robot_prim_path)
            robot_found = True
            print("Loaded robot using isaacsim.robots.robot.Robot")
    except Exception as e1:
        print(f"Method 1 failed: {e1}")
        
        # Method 2: Try using omni.isaac.franka.Franka
        try:
            franka_module = try_import("omni.isaac.franka")
            if franka_module:
                my_franka = franka_module.Franka(prim_path=robot_prim_path, name="franka_robot")
                my_world.scene.add(my_franka)
                robot_found = True
                print("Loaded robot using omni.isaac.franka.Franka")
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            
            # Method 3: Try using articulation directly
            try:
                articulation_module = try_import("isaacsim.scene.articulations")
                if articulation_module:
                    my_franka = articulation_module.Articulation(prim_path=robot_prim_path)
                    my_world.scene.add(my_franka)
                    robot_found = True
                    print("Loaded robot using isaacsim.scene.articulations.Articulation")
                else:
                    # Method 4: Try omni.isaac.core.articulations
                    articulation_module = try_import("omni.isaac.core.articulations")
                    if articulation_module:
                        my_franka = articulation_module.Articulation(prim_path=robot_prim_path)
                        my_world.scene.add(my_franka)
                        robot_found = True
                        print("Loaded robot using omni.isaac.core.articulations.Articulation")
            except Exception as e3:
                print(f"Method 3/4 failed: {e3}")
    
    # Method 5: Try getting the robot directly from the scene after the world has been stepped
    if not robot_found:
        try:
            # Step the world to let it recognize objects
            for _ in range(5):
                my_world.step(render=True)
            
            # Try to get the robot by name or by path
            try:
                my_franka = my_world.scene.get_object("franka")
                robot_found = True
                print("Found robot using scene.get_object('franka')")
            except:
                try:
                    # Try getting all objects and finding the one with 'franka' in the name
                    for obj_name in my_world.scene.get_object_names():
                        if "franka" in obj_name.lower():
                            my_franka = my_world.scene.get_object(obj_name)
                            robot_found = True
                            print(f"Found robot using scene.get_object('{obj_name}')")
                            break
                except Exception as e:
                    print(f"Method 5 failed: {e}")
        except Exception as e:
            print(f"Error stepping world: {e}")
    
    if not my_franka:
        print("ERROR: Could not load the Franka robot at path '/World/franka'.")
        print("Make sure the robot exists and the Isaac Sim environment is properly set up.")
        simulation_app.close()
        return
      # Initialize the robot movements controller
    print("Initializing robot movements controller...")
    
    # Make sure the robot is properly loaded into the scene
    simulation_app.update()
    for _ in range(10):  # Give the scene some steps to fully initialize
        my_world.step(render=True)
    
    # Create the robot movements controller
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


def run_random_joint_movement_simple():
    """
    Run Isaac Sim simulation with a Franka robot and demonstrate
    random joint movements without using the PickPlace task.
    Uses the known path "/World/franka" directly.
    """
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})
    
    # Basic imports
    import numpy as np
    import importlib
    import sys
    
    # Dynamically import required modules
    def try_import(module_name):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            return None
    
    # Import omni modules - this should work on all versions
    omni = try_import("omni")
    if not omni:
        print("ERROR: Could not import omni module. Make sure Isaac Sim is properly set up.")
        simulation_app.close()
        return
    
    # Open the scene
    omni.usd.get_context().open_stage(str(THIS_FOLDER / "scenes" / "franka_blocks_manual.usd"))
    simulation_app.update()
    
    # Get the stage
    stage = omni.usd.get_context().get_stage()
    
    # Check for the known Franka robot path
    robot_prim_path = "/World/franka"
    if not stage.GetPrimAtPath(robot_prim_path):
        print(f"ERROR: No robot found at {robot_prim_path}")
        print("Make sure the Franka robot exists in the USD file and try again.")
        simulation_app.close()
        return
    
    print(f"Found Franka robot at: {robot_prim_path}")
    
    # Try to detect the Isaac Sim API version (newer or older)
    isaacsim_core = try_import("isaacsim.core.api")
    
    # Create the world and load the robot
    my_franka = None
    
    if isaacsim_core:
        # Newer API (2023+)
        print("Using newer isaacsim API")
        
        # Create the world
        my_world = isaacsim_core.World(stage_units_in_meters=1.0)
        my_world.reset()
        
        # First try the direct approach - add an articulation directly
        articulation_module = try_import("isaacsim.scene.articulations")
        if articulation_module:
            try:
                my_franka = articulation_module.Articulation(prim_path=robot_prim_path)
                my_world.scene.add(my_franka)
                print("Loaded robot using isaacsim.scene.articulations.Articulation")
            except Exception as e:
                print(f"Error loading articulation: {e}")
                my_franka = None
          # If that didn't work, try other approaches
        if not my_franka:
            # Step the world a few times to initialize objects
            for _ in range(10):
                my_world.step(render=True)
                simulation_app.update()
            
            # Try to get the object directly from the scene
            try:
                my_franka = my_world.scene.get_object("franka")
                if my_franka:
                    print("Found robot using scene.get_object('franka')")
            except Exception as e:
                print(f"Error getting robot from scene by name: {e}")
                
            # If we still don't have the robot, try to get all objects and find one with franka in the name
            if not my_franka:
                try:
                    print("Trying to get all objects in the scene...")
                    object_names = my_world.scene.get_object_names()
                    print(f"Objects in scene: {object_names}")
                    
                    for obj_name in object_names:
                        if "franka" in obj_name.lower():
                            my_franka = my_world.scene.get_object(obj_name)
                            print(f"Found robot using scene.get_object('{obj_name}')")
                            break
                except Exception as e:
                    print(f"Error getting objects from scene: {e}")
                    my_franka = None
        
    else:
        # Older API (pre-2023)
        print("Using older omni.isaac API")
        from omni.isaac.core.world import World
        
        # Create the world using older API
        my_world = World()
        my_world.reset()
        
        # Try using articulation directly
        try:
            from omni.isaac.core.articulations import Articulation
            my_franka = Articulation(prim_path=robot_prim_path)
            my_world.scene.add(my_franka)
            print("Loaded robot using omni.isaac.core.articulations.Articulation")
        except Exception as e:
            print(f"Error loading articulation: {e}")
            
            # Try using Franka-specific class if available
            try:
                from omni.isaac.franka import Franka
                my_franka = Franka(prim_path=robot_prim_path, name="franka_robot")
                my_world.scene.add(my_franka)
                print("Loaded robot using omni.isaac.franka.Franka")
            except Exception as e:
                print(f"Error loading Franka: {e}")
                my_franka = None
      # If we still couldn't load the robot, report error and try to provide more information
    if not my_franka:
        print("ERROR: Could not load the Franka robot at path '/World/franka'.")
        print("Make sure the robot exists and the Isaac Sim environment is properly set up.")
        
        # Try to get more diagnostic information
        try:
            print("\nDiagnostic information:")
            print("1. Checking if robot prim exists again:")
            if stage.GetPrimAtPath(robot_prim_path):
                prim = stage.GetPrimAtPath(robot_prim_path)
                prim_type = prim.GetTypeName()
                print(f"  - Robot prim exists with type: {prim_type}")
                print(f"  - Prim path: {robot_prim_path}")
            else:
                print("  - Robot prim does not exist at path:", robot_prim_path)
            
            print("2. Available objects in the scene:")
            try:
                object_names = my_world.scene.get_object_names()
                if object_names:
                    print(f"  - Objects: {object_names}")
                else:
                    print("  - No objects found in the scene")
            except Exception as e:
                print(f"  - Error getting objects: {e}")
            
            print("3. Try direct access to the articulation controller:")
            try:
                import omni
                xform = stage.GetPrimAtPath(robot_prim_path)
                if xform:
                    print(f"  - Robot prim exists, trying to get its properties")
                    # Print some properties of the prim
                    for prop in xform.GetPropertyNames():
                        print(f"    - Property: {prop}")
            except Exception as e:
                print(f"  - Error accessing prim properties: {e}")
        except Exception as e:
            print(f"Error during diagnostics: {e}")
        
        simulation_app.close()
        return
    
    # Make sure the robot is properly loaded into the scene
    print("Initializing the scene...")
    simulation_app.update()
    for _ in range(10):  # Give the scene some steps to fully initialize
        my_world.step(render=True)
    
    # Create the robot movements controller
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
    # Use the simplified version that directly accesses the robot at "/World/franka"
    run_random_joint_movement_simple()
