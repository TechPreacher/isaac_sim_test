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
    random joint movements.
    """
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})

    import numpy as np
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.robot.manipulators.examples.franka.tasks import PickPlace

    # Open the scene
    omni.usd.get_context().open_stage(str(THIS_FOLDER / "scenes" / "franka_blocks_manual.usd"))
    simulation_app.update()

    # Set up the world
    my_world = World(stage_units_in_meters=1.0)
    my_task = PickPlace()
    my_world.add_task(my_task)
    my_world.reset()

    # Get the robot
    task_params = my_task.get_params()
    robot_name = task_params["robot_name"]["value"]
    my_franka = my_world.scene.get_object(robot_name)
    
      # Initialize the robot movements controller
    robot_movements = RobotMovements(my_franka, max_velocity=0.2)  # Lower velocity for slower movements
    
    # Run the simulation
    reset_needed = False
    movement_count = 0
    max_movements = 10  # Number of random movements to perform
    movement_interval = 200  # Increased from 100 to 200 for slower movements
    
    print("Starting simulation with random joint movements...")
    print(f"Robot has {len(my_franka.dof_names)} joints: {my_franka.dof_names}")
    
    while simulation_app.is_running() and movement_count < max_movements:
        my_world.step(render=True)
        
        if my_world.is_stopped() and not reset_needed:
            reset_needed = True
            
        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                reset_needed = False
                  # Every 200 steps, set random joint positions (increased from 100)
            if my_world.current_time_step_index % movement_interval == 0 and movement_count < max_movements:
                print(f"\nMovement {movement_count + 1}/{max_movements}")
                random_positions = robot_movements.set_random_joint_positions(scale_factor=0.4)
                print(f"Random joint positions: {random_positions}")
                movement_count += 1
                
                # Wait for a moment to see the effect (increased wait time)
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
    run_random_joint_movement()
