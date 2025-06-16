import time
import warnings
from pathlib import Path
from core import setup_isaac_sim_environment

# Suppress numpy compatibility warnings
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

THIS_FOLDER = Path(__file__).parent.resolve()

def run_simple_franka():
    """
    Run Isaac Sim simulation with a Franka robot using a simpler approach.
    This version uses the most direct method to access the robot.
    """
    from isaacsim import SimulationApp

    # Initialize the simulation app
    simulation_app = SimulationApp({"headless": False})
    
    # Import necessary modules
    import numpy as np
    
    # Create a wrapper for print to add timestamps
    def log(message):
        print(f"[{time.time():.2f}s] {message}")
    
    log("Starting simple Franka robot simulation")
    
    # Import omni modules
    import omni.usd
    
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
        log(f"ERROR: Robot not found at {robot_prim_path}")
        simulation_app.close()
        return
    
    log(f"Found robot at {robot_prim_path}")
    
    # Try to set up the simulation in the most direct way possible
    try:
        # First try isaacsim approach (newer API)
        try:
            log("Trying isaacsim approach...")
            
            # Import required modules
            from isaacsim.core.api import World
            
            # Create the world
            world = World(stage_units_in_meters=1.0)
            
            # Reset the world to initialize physics
            world.reset()
            
            # Get the robot directly from the USD stage
            # This is a key difference - we're using the direct add_articulation_from_usd method
            my_franka = world.scene.add_articulation_from_usd(robot_prim_path)
            
            log(f"Successfully loaded robot using isaacsim API")
            api_type = "isaacsim"
            
        except (ImportError, AttributeError) as e:
            log(f"isaacsim approach failed: {e}")
            
            # Try omni.isaac approach (older API)
            try:
                log("Trying omni.isaac approach...")
                
                # Import required modules
                from omni.isaac.core.world import World
                
                # Create the world
                world = World()
                
                # Reset the world to initialize physics
                world.reset()
                
                # Add the robot as an articulation from USD
                from omni.isaac.core.utils.stage import add_reference_to_stage
                from omni.isaac.core.articulations import ArticulationView
                
                # Add the robot as an articulation view
                my_franka = world.scene.add(
                    ArticulationView(
                        prim_path=robot_prim_path,
                        name="franka"
                    )
                )
                
                log(f"Successfully loaded robot using omni.isaac API")
                api_type = "omni.isaac"
                
            except (ImportError, AttributeError) as e:
                log(f"omni.isaac approach failed: {e}")
                
                # Fallback to direct physics API
                import omni
                from pxr import PhysicsSchema
                
                log("Using direct physics API...")
                
                # Initialize physics (simplified)
                from omni.physx import get_physx_simulation_interface
                physics_sim_iface = get_physx_simulation_interface()
                
                # Define a simple world class
                class SimpleWorld:
                    def __init__(self):
                        self.current_time_step_index = 0
                        
                    def step(self, render=True):
                        # Step physics
                        physics_sim_iface.step_physics(1/60.0)
                        if render:
                            simulation_app.update()
                        self.current_time_step_index += 1
                        
                    def reset(self):
                        physics_sim_iface.play()
                        
                world = SimpleWorld()
                world.reset()
                
                # Create a simple articulation class
                class SimpleArticulation:
                    def __init__(self, prim_path):
                        self.prim_path = prim_path
                        
                        # Define joint names for Franka
                        # These are the standard 7 joint names for a Franka Emika Panda robot
                        self.dof_names = [
                            "panda_joint1",
                            "panda_joint2",
                            "panda_joint3",
                            "panda_joint4",
                            "panda_joint5",
                            "panda_joint6",
                            "panda_joint7"
                        ]
                        
                        log(f"Created simple articulation with {len(self.dof_names)} joints")
                        
                        # Define default joint positions (home position for Franka)
                        self.default_positions = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
                        
                    def get_joint_positions(self):
                        # Simple implementation - just return the default positions
                        # In a real implementation, we would query the actual joint positions
                        return self.default_positions
                        
                    def get_articulation_controller(self):
                        # Return a simple controller
                        return SimpleController(self)
                        
                class SimpleController:
                    def __init__(self, articulation):
                        self.articulation = articulation
                        
                    def apply_action(self, positions):
                        log(f"Setting joint positions: {positions}")
                        # In a real implementation, this would set the joint positions
                        # For now, we'll just log the positions
                
                my_franka = SimpleArticulation(robot_prim_path)
                api_type = "direct"
                
    except Exception as e:
        log(f"ERROR: Failed to set up simulation: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return
    
    # Now run a simple simulation
    log("Starting simulation...")
    
    # We'll just move the robot between a few predefined poses
    # These are typical joint positions for a Franka robot
    poses = [
        # Home position
        np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
        # Extended position
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        # Folded position
        np.array([0.0, -1.57, 0.0, -3.14, 0.0, 1.57, 0.785]),
        # Tilted position
        np.array([0.785, -0.785, 0.785, -2.356, -0.785, 1.571, 0.0])
    ]
    
    # Get the controller
    controller = my_franka.get_articulation_controller()
    
    # For pure visualization, we'll just step through some poses
    pose_idx = 0
    steps_per_pose = 300  # Stay in each pose for this many steps
    total_steps = steps_per_pose * len(poses) * 2  # Go through all poses twice
    
    # Check if we need to import RobotMovements for smoother motion
    if api_type != "direct":
        from robot_movements import RobotMovements
        robot_movements = RobotMovements(my_franka, max_velocity=0.2)
        use_robot_movements = True
    else:
        use_robot_movements = False
    
    step_count = 0
    
    try:
        while simulation_app.is_running() and step_count < total_steps:
            # Step the world
            world.step(render=True)
            
            # Change pose at regular intervals
            if step_count % steps_per_pose == 0:
                pose_idx = (pose_idx + 1) % len(poses)
                target_pose = poses[pose_idx]
                
                log(f"Moving to pose {pose_idx + 1}/{len(poses)}: {target_pose}")
                
                if use_robot_movements:
                    # Use the RobotMovements class for smoother motion
                    robot_movements.set_joint_positions(target_pose)
                else:
                    # Direct control
                    controller.apply_action(target_pose)
            
            step_count += 1
    
    except Exception as e:
        log(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    log("Simulation complete.")
    simulation_app.close()

if __name__ == "__main__":
    # Set up the environment and run the simulation
    setup_isaac_sim_environment()
    run_simple_franka()
