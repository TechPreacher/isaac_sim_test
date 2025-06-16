import time
import warnings
from pathlib import Path
from core import setup_isaac_sim_environment

# Suppress numpy compatibility warnings
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

THIS_FOLDER = Path(__file__).parent.resolve()

def run_franka_with_control():
    """
    Run Isaac Sim simulation with a Franka robot and control its joints.
    This builds on the minimal_franka.py approach but adds joint control.
    """
    # Import SimulationApp first to initialize the simulation
    from isaacsim import SimulationApp

    # Create the simulation app
    simulation_app = SimulationApp({"headless": False})
    
    # Import the basic modules
    import time
    import numpy as np
    import omni.usd
    import omni.timeline
    
    # Create a helper function for logging with timestamps
    def log(message):
        print(f"[{time.time():.2f}s] {message}")
    
    log("Starting Franka robot simulation with joint control")
    
    # Open the USD stage with the scene
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
    
    # Get timeline interface for controlling playback
    timeline = omni.timeline.get_timeline_interface()
    
    # Start the simulation
    timeline.play()
    simulation_app.update()
    
    # Try different methods to access the robot articulation
    log("Trying to access the robot as an articulation...")
    
    # Wait for physics to initialize
    for i in range(10):
        simulation_app.update()
        time.sleep(0.1)
        
    robot = None
    joint_positions = None
    
    # Method 1: Try using omni.isaac.core.articulations.Articulation
    try:
        log("Method 1: Using omni.isaac.core.articulations.Articulation")
        from omni.isaac.core.articulations import Articulation
        
        # Create articulation from existing prim
        robot = Articulation(prim_path=robot_prim_path, name="franka")
        
        # Update app to ensure articulation is registered
        simulation_app.update()
        time.sleep(0.2)
        
        # Initialize the robot
        robot.initialize()
        log("Successfully initialized the robot articulation!")
        
        # Get joint information
        dof_names = robot.dof_names
        dof_count = robot.num_dof
        log(f"Robot has {dof_count} degrees of freedom")
        log(f"Joint names: {dof_names}")
        
        # Get current joint positions
        joint_positions = robot.get_joint_positions()
        log(f"Current joint positions: {joint_positions}")
        
    except Exception as e:
        log(f"Method 1 failed: {e}")
        robot = None
    
    # Method 2: Try direct physics API if Method 1 failed
    if robot is None:
        try:
            log("Method 2: Using physics API directly")
            import omni.physics
            from pxr import PhysicsSchemaTools
            
            # Get physics scene
            physics_context = omni.physics.get_physics_context()
            physics_scene = physics_context.scene
            
            # Ensure physics is initialized
            simulation_app.update()
            time.sleep(0.2)
            
            # Try to get articulation view
            log("Looking for articulations in the scene...")
            
            # Check if the robot prim has physics articulation properties
            log("Checking robot prim properties:")
            for prop_name in robot_prim.GetPropertyNames():
                if "physics" in prop_name.lower() or "articulation" in prop_name.lower():
                    try:
                        property_value = robot_prim.GetProperty(prop_name).Get()
                        log(f"  - {prop_name}: {property_value}")
                    except:
                        log(f"  - {prop_name}: <unable to read value>")
            
            # Try using Omnigraph-based access
            try:
                from omni.physx import get_physx_interface
                physx = get_physx_interface()
                articulation_count = physx.get_articulation_count()
                log(f"Found {articulation_count} articulations via PhysX interface")
                
                if articulation_count > 0:
                    # Try to identify our robot
                    for i in range(articulation_count):
                        # This is version-dependent, so wrap in try-except
                        try:
                            articulation_handle = physx.get_articulation_handle(i)
                            log(f"Articulation {i} handle: {articulation_handle}")
                            # If we could identify our specific robot, we would do it here
                        except Exception as e:
                            log(f"Could not get handle for articulation {i}: {e}")
            except Exception as e:
                log(f"Error accessing PhysX interface: {e}")
            
        except Exception as e:
            log(f"Method 2 failed: {e}")
    
    # Run the simulation with control if we have a robot
    total_steps = 1000
    step_count = 0
    
    if robot is not None and joint_positions is not None:
        log("Running simulation with joint control...")
        
        try:
            # Simple oscillation pattern for joints
            while simulation_app.is_running() and step_count < total_steps:
                # Calculate sine wave position for first joint
                t = step_count / 50.0  # time parameter
                new_positions = joint_positions.copy()
                new_positions[0] = 0.5 * np.sin(t)  # Simple oscillation for first joint
                
                # Apply joint positions
                robot.set_joint_positions(new_positions)
                
                # Update simulation
                simulation_app.update()
                
                # Print status every 100 steps
                if step_count % 100 == 0:
                    log(f"Step {step_count}/{total_steps}, Joint pos: {new_positions[0]:.2f}")
                
                step_count += 1
                time.sleep(0.01)
                
        except Exception as e:
            log(f"Error during joint control: {e}")
            import traceback
            traceback.print_exc()
    else:
        log("Running simulation without joint control (visualization only)...")
        
        try:
            while simulation_app.is_running() and step_count < total_steps:
                # Update the app to step the simulation
                simulation_app.update()
                
                # Print status every 100 steps
                if step_count % 100 == 0:
                    log(f"Step {step_count}/{total_steps}")
                
                step_count += 1
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
    run_franka_with_control()
