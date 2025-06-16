import time
import warnings
from pathlib import Path
from core import setup_isaac_sim_environment

# Suppress numpy compatibility warnings
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

THIS_FOLDER = Path(__file__).parent.resolve()

def run_minimal_franka():
    """
    Run Isaac Sim simulation with a Franka robot using the most minimal approach possible.
    This version uses the absolute minimum code needed to open the scene and view the robot.
    """
    # Import SimulationApp first to initialize the simulation
    from isaacsim import SimulationApp

    # Create the simulation app
    simulation_app = SimulationApp({"headless": False})
    
    # Import the basic modules
    import time
    import numpy as np
    import omni.usd
    
    # Create a helper function for logging with timestamps
    def log(message):
        print(f"[{time.time():.2f}s] {message}")
    
    log("Starting minimal Franka robot simulation")
    
    # Open the USD stage with the scene - make sure to get the stage correctly
    usd_context = omni.usd.get_context()
    usd_context.open_stage(str(THIS_FOLDER / "scenes" / "franka_blocks_manual.usd"))
    simulation_app.update()
    
    # Get the stage after opening it
    stage = usd_context.get_stage()
    
    # Define the robot path
    robot_prim_path = "/World/franka"
    
    # Check if the robot exists in the stage
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim:
        log(f"ERROR: Robot not found at {robot_prim_path}")
        simulation_app.close()
        return
    
    log(f"Found robot prim at {robot_prim_path} with type: {robot_prim.GetTypeName()}")
    
    # Import timeline for simulation control
    import omni.timeline
    
    # Get the timeline interface for controlling playback
    timeline = omni.timeline.get_timeline_interface()
    
    # Start the simulation
    timeline.play()
    
    # Run for a fixed number of steps
    total_steps = 1000
    step_count = 0
    
    log("Starting simulation - just viewing the robot without manipulation")
    
    try:
        while simulation_app.is_running() and step_count < total_steps:
            # Update the app to step the simulation
            simulation_app.update()
            
            # Print status every 100 steps
            if step_count % 100 == 0:
                log(f"Step {step_count}/{total_steps}")
            
            step_count += 1
            
            # Add a small sleep to control simulation speed
            time.sleep(0.01)
    
    except Exception as e:
        log(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    log("Simulation complete")
    simulation_app.close()

if __name__ == "__main__":
    # Set up the environment and run the simulation
    setup_isaac_sim_environment()
    run_minimal_franka()
