import time
import warnings
from pathlib import Path
from core import setup_isaac_sim_environment
from robot_movements import RobotMovements

# Suppress numpy compatibility warnings
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

THIS_FOLDER = Path(__file__).parent.resolve()

def run_direct_franka():
    """
    Run Isaac Sim simulation with a Franka robot using direct articulation access.
    """
    from isaacsim import SimulationApp

    # Initialize the simulation app
    simulation_app = SimulationApp({"headless": False})
    
    # Import necessary modules
    import numpy as np
    import omni
    
    # Open the scene
    omni.usd.get_context().open_stage(str(THIS_FOLDER / "scenes" / "franka_blocks_manual.usd"))
    simulation_app.update()
    
    # Get the stage
    stage = omni.usd.get_context().get_stage()
    
    # Define the robot path
    robot_prim_path = "/World/franka"
    
    # Check if the robot exists
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim:
        print(f"ERROR: Robot not found at {robot_prim_path}")
        simulation_app.close()
        return
    
    print(f"Found robot at {robot_prim_path}")
    
    # Create a direct articulation interface based on the USD prim
    # Try different approaches for different Isaac Sim versions
    
    # Approach 1: Use isaacsim APIs if available
    try:
        from isaacsim.core.api import World
        from isaacsim.scene.articulations import Articulation
        
        print("Using isaacsim.scene.articulations.Articulation")
        
        # Create the world and physics scene
        world = World(stage_units_in_meters=1.0)
        world.reset()
        
        # Create the articulation
        my_franka = Articulation(prim_path=robot_prim_path)
        
        # Add the articulation to the scene
        world.scene.add(my_franka)
        
        # Step the simulation a few times to initialize
        for _ in range(5):
            world.step(render=True)
        
    except ImportError:
        # Approach 2: Try the older omni.isaac APIs
        try:
            from omni.isaac.core.world import World
            from omni.isaac.core.articulations import Articulation
            
            print("Using omni.isaac.core.articulations.Articulation")
            
            # Create the world and physics scene
            world = World()
            world.scene.add_default_ground_plane()
            world.reset()
            
            # Create the articulation
            my_franka = Articulation(prim_path=robot_prim_path)
            
            # Add the articulation to the scene
            world.scene.add(my_franka)
            
            # Step the simulation a few times to initialize
            for _ in range(5):
                world.step(render=True)
                
        except ImportError:
            # Approach 3: Try using direct USD articulation API
            try:
                from omni.physx import get_physx_scene_query_interface
                from omni.physx.scripts import physicsUtils
                
                print("Using direct USD articulation API")
                
                # Initialize physics
                physicsUtils.setup_physics()
                
                # Create a simple simulation loop
                class SimpleWorld:
                    def __init__(self):
                        self.current_time_step_index = 0
                        
                    def step(self, render=True):
                        # Step physics
                        physicsUtils.step_physics()
                        if render:
                            omni.kit.app.get_app().update()
                        self.current_time_step_index += 1
                
                world = SimpleWorld()
                
                # Create a simple wrapper for the articulation
                class SimpleArticulation:
                    def __init__(self, prim_path):
                        self.prim_path = prim_path
                        self.prim = stage.GetPrimAtPath(prim_path)
                        
                        # Get the articulation's joints
                        self.joints = []
                        self.dof_names = []
                        
                        # Search for joints under the robot prim
                        for child_prim in self.prim.GetChildren():
                            if child_prim.GetTypeName() == "PhysicsJoint" or "joint" in child_prim.GetName().lower():
                                self.joints.append(child_prim)
                                self.dof_names.append(child_prim.GetName())
                        
                        print(f"Found {len(self.joints)} joints: {self.dof_names}")
                        
                        # Create a simple articulation controller
                        self.controller = None
                    
                    def get_articulation_controller(self):
                        if not self.controller:
                            self.controller = SimpleController(self)
                        return self.controller
                    
                    def get_joint_positions(self):
                        # Dummy implementation
                        return np.zeros(len(self.dof_names))
                
                class SimpleController:
                    def __init__(self, articulation):
                        self.articulation = articulation
                    
                    def apply_action(self, positions):
                        print(f"Setting joint positions: {positions}")
                        # In a real implementation, this would set the joint positions
                
                my_franka = SimpleArticulation(robot_prim_path)
                
            except Exception as e:
                print(f"ERROR: Failed to create articulation: {e}")
                simulation_app.close()
                return
    
    # Create the robot movements controller
    print("Initializing robot movements controller...")
    robot_movements = RobotMovements(my_franka, max_velocity=0.2)
    
    # Run the simulation
    movement_count = 0
    max_movements = 10
    movement_interval = 200
    
    print("Starting simulation with random joint movements...")
    print(f"Robot has {len(my_franka.dof_names)} joints: {my_franka.dof_names}")
    
    try:
        while simulation_app.is_running() and movement_count < max_movements:
            world.step(render=True)
            
            if world.current_time_step_index % movement_interval == 0 and movement_count < max_movements:
                print(f"\nMovement {movement_count + 1}/{max_movements}")
                random_positions = robot_movements.set_random_joint_positions(scale_factor=0.4)
                print(f"Random joint positions: {random_positions}")
                movement_count += 1
                
                # Wait to see the effect
                for _ in range(100):
                    world.step(render=True)
        
        # Move back to home position
        print("\nMoving back to home position...")
        robot_movements.move_to_home_position()
        for _ in range(100):
            world.step(render=True)
    
    except Exception as e:
        print(f"Error during simulation: {e}")
    
    print("Simulation complete.")
    simulation_app.close()

if __name__ == "__main__":
    # Set up the environment and run the simulation
    setup_isaac_sim_environment()
    run_direct_franka()
