import time
import warnings
from pathlib import Path
from core import setup_isaac_sim_environment

# Suppress numpy compatibility warnings
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

THIS_FOLDER = Path(__file__).parent.resolve()

def run_bare_franka():
    """
    Run Isaac Sim simulation with a Franka robot using the most basic approach.
    This version avoids using any advanced APIs and directly uses the most
    basic methods available in Isaac Sim.
    """
    # Import SimulationApp first to initialize the simulation
    from isaacsim import SimulationApp
    
    # Create the simulation app
    simulation_app = SimulationApp({"headless": False})
    
    # Import the basic modules
    import time
    import numpy as np
    
    # Create a helper function for logging with timestamps
    def log(message):
        print(f"[{time.time():.2f}s] {message}")
    
    log("Starting simplified Franka robot simulation")
    
    # Import omni modules that should be available after SimulationApp initialization
    import omni
    
    # Open the USD stage with the scene
    stage = omni.usd.get_context().open_stage(str(THIS_FOLDER / "scenes" / "franka_blocks_manual.usd"))
    simulation_app.update()
    
    # Define the robot path
    robot_prim_path = "/World/franka"
    
    # Check if the robot exists in the stage
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim:
        log(f"ERROR: Robot not found at {robot_prim_path}")
        simulation_app.close()
        return
    
    log(f"Found robot prim at {robot_prim_path}")
    
    # Try to detect which API is available
    world = None
    
    # Method 1: Try isaacsim.core.api.World
    try:
        from isaacsim.core.api import World
        world = World(stage_units_in_meters=1.0)
        log("Using isaacsim.core.api.World")
        api_type = "isaacsim"
    except ImportError:
        log("isaacsim.core.api.World not available")
        api_type = None
    
    # Method 2: Try omni.isaac.core.world.World
    if not world:
        try:
            from omni.isaac.core.world import World
            world = World()
            log("Using omni.isaac.core.world.World")
            api_type = "omni.isaac"
        except ImportError:
            log("omni.isaac.core.world.World not available")
    
    # Method 3: Create a minimal world
    if not world:
        log("Creating a minimal world implementation")
        api_type = "minimal"
        
        # Create a minimal world implementation
        class MinimalWorld:
            def __init__(self):
                self.current_time_step_index = 0
                
                # Initialize physics
                self._timeline = omni.timeline.get_timeline_interface()
                self._physx = omni.physx.acquire_physx_interface()
                
                # Start physics
                self._timeline.play()
                
            def step(self, render=True):
                if render:
                    simulation_app.update()
                self.current_time_step_index += 1
                
            def reset(self):
                self._timeline.stop()
                self._timeline.play()
        
        world = MinimalWorld()
    
    # Reset the world to initialize physics
    world.reset()
    simulation_app.update()
    
    log("Physics initialized")
    
    # Define joint positions for demonstration
    home_position = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    
    # Just run the simulation and visualize the robot
    log("Starting visualization...")
    
    # Run for a fixed number of steps
    total_steps = 1000
    step_count = 0
    
    try:
        while simulation_app.is_running() and step_count < total_steps:
            # Step the world
            world.step(render=True)
            
            # Print status every 100 steps
            if step_count % 100 == 0:
                log(f"Step {step_count}/{total_steps}")
            
            step_count += 1
    
    except Exception as e:
        log(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    log("Simulation complete")
    simulation_app.close()

if __name__ == "__main__":
    # Set up the environment and run the simulation
    setup_isaac_sim_environment()
    run_bare_franka()
