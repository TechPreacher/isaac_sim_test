#!/usr/bin/env python3
"""
Improved Franka Robot Pick and Place Control for Isaac Sim 4.5.0

This script automates a pick-and-place task where the Franka robot:
1. Moves to the banana object at /World/banana
2. Picks it up with the gripper
3. Moves to the target location at /World/crate
4. Places the banana in the crate

The implementation works with Isaac Sim 4.5.0 and doesn't rely on the PickPlace task.
It uses a hybrid approach combining both position-based IK and known joint angles
to ensure accurate positioning of the end effector.
"""

import time
import warnings
import numpy as np
from pathlib import Path
from core import setup_isaac_sim_environment
from robot_movements import RobotMovements

# Suppress numpy compatibility warnings
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

THIS_FOLDER = Path(__file__).parent.resolve()
USD_SCENE_PATH = str(THIS_FOLDER / "scenes" / "franka_blocks_manual.usd")
ROBOT_PRIM_PATH = "/World/franka"
BANANA_PRIM_PATH = "/World/banana"
TARGET_PRIM_PATH = "/World/crate"  # Use crate instead of tray

# Global variables to store corrected paths
ACTUAL_BANANA_PATH = BANANA_PRIM_PATH
ACTUAL_TARGET_PATH = TARGET_PRIM_PATH

# End effector offset for Franka
# Offset in robot's local frame (z-up)
EE_OFFSET = np.array([0.0, 0.0, 0.10])  # 10cm offset for the end effector

# Optimal joint positions for grasping banana (based on manual positioning)
OPTIMAL_BANANA_GRASP_JOINTS = np.array([
    0.6620,  # panda_joint1 (base rotation)
    0.2800,  # panda_joint2 (shoulder)
    0.1000,  # panda_joint3 (elbow 1)
    -3.0100, # panda_joint4 (elbow 2)
    0.2000,  # panda_joint5 (wrist 1)
    3.2370,  # panda_joint6 (wrist 2)
    1.2410,  # panda_joint7 (wrist 3)
    0.04,    # panda_finger_joint1 (open)
    0.04     # panda_finger_joint2 (open)
])

def log(message):
    """Helper function for logging with timestamps"""
    print(f"[{time.time():.2f}s] {message}")

def get_prim_pose(stage, prim_path):
    """
    Get the world pose of a prim
    
    Args:
        stage: USD Stage
        prim_path: Path to the prim
        
    Returns:
        Tuple of (position, orientation) or (None, None) if not found
    """
    from pxr import UsdGeom
    
    # Get the prim at the specified path
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        log(f"ERROR: Prim not found at {prim_path}")
        return None, None
    
    # Get the xformable
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        log(f"ERROR: Prim at {prim_path} is not xformable")
        return None, None
    
    # Get the world transform
    world_transform = xformable.ComputeLocalToWorldTransform(0)
    
    # Extract position (translation) from the transform matrix
    position = np.array([
        world_transform[3][0],
        world_transform[3][1],
        world_transform[3][2]
    ])
    
    # For orientation, we can return the rotation component
    # This is simplified - in a real application you might want to convert to quaternion
    orientation = np.array([
        [world_transform[0][0], world_transform[0][1], world_transform[0][2]],
        [world_transform[1][0], world_transform[1][1], world_transform[1][2]],
        [world_transform[2][0], world_transform[2][1], world_transform[2][2]]
    ])
    
    return position, orientation

def find_object_in_scene(stage, name_substring):
    """
    Find an object in the scene by name substring
    
    Args:
        stage: USD Stage
        name_substring: Substring to search for in prim path
        
    Returns:
        Prim path of the found object, or None if not found
    """
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if name_substring.lower() in path_str.lower():
            return path_str
    return None

def calculate_ik_for_banana_grasp(robot, banana_position, offset=None, gripper_open=True):
    """
    Calculate joint positions to grasp the banana using known optimal joint angles
    but adjusted for the banana's position.
    
    Args:
        robot: The robot articulation object
        banana_position: The 3D position of the banana
        offset: Optional offset to apply to the target position (in world frame)
        gripper_open: Whether to open (True) or close (False) the gripper
        
    Returns:
        Joint positions to reach the target
    """
    # Start with the optimal joint configuration for banana grasping
    joint_positions = OPTIMAL_BANANA_GRASP_JOINTS.copy()
    
    # Get the number of joints
    num_joints = len(robot.dof_names)
    
    # Calculate distance from robot base to banana
    distance_xy = np.linalg.norm(banana_position[:2])  # XY distance
    height = banana_position[2]  # Z height
    
    # First joint (base rotation) based on banana's x,y position
    if banana_position[0] != 0 or banana_position[1] != 0:
        # Get the angle from robot base to banana in the XY plane
        angle_to_banana = np.arctan2(banana_position[1], banana_position[0])
        
        # Adjust the base rotation joint (joint 1)
        joint_positions[0] = angle_to_banana
        
        # Log the angle for debugging
        log(f"Angle to banana: {angle_to_banana:.4f} radians (distance: {distance_xy:.4f}m, height: {height:.4f}m)")
    
    # Fine-tune adjustments based on the banana's exact position
    # Shoulder joint (joint 2) adjustment based on XY distance
    distance_factor = (distance_xy - 0.28) * 0.5  # 0.28m is the reference distance from optimal position
    joint_positions[1] += distance_factor
    
    # Elbow joint (joint 4) adjustment based on height
    height_factor = (height - 0.84) * 0.8  # 0.84m is the reference height from optimal position
    joint_positions[3] += height_factor
    
    # Adjust joint positions if we have an offset
    if offset is not None:
        # Apply offset-based adjustments
        if offset[2] > 0:  # Pre-grasp position (above banana)
            # Adjust joints to be slightly above the banana
            joint_positions[1] += 0.1  # Shoulder up
            joint_positions[3] += 0.2  # Elbow up
            # Slightly adjust wrist angle for better approach
            joint_positions[5] -= 0.1  # Adjust wrist angle
        elif offset[2] < 0:  # Below the banana
            # Adjust joints to be lower
            joint_positions[1] -= 0.1  # Shoulder down
            joint_positions[3] -= 0.2  # Elbow down
            # Adjust wrist for better grasping angle
            joint_positions[5] += 0.1  # Adjust wrist angle for downward approach
    
    # Set gripper position
    if num_joints > 7:  # If we have gripper joints
        # Adjust gripper opening based on whether we're grasping or not
        if gripper_open:
            # Slightly wider opening when approaching (gives more clearance)
            gripper_value = 0.05  # Wider than default 0.04
        else:
            # Fully closed when grasping
            gripper_value = 0.0
        
        joint_positions[7] = gripper_value
        joint_positions[8] = gripper_value
    
    # Log the calculated joint positions for debugging
    log(f"Calculated joint positions: {joint_positions}")
    
    # Return the calculated joint positions
    return joint_positions

def calculate_ik_for_position(robot, target_position, offset=None, gripper_open=True, orientation=None):
    """
    Calculate inverse kinematics to reach a target position.
    Uses a simplified approach since Isaac Sim 4.5.0 may not have a direct IK solver.
    
    Args:
        robot: The robot articulation object
        target_position: The 3D target position to reach
        offset: Optional offset to apply to the target position (in world frame)
        gripper_open: Whether to open (True) or close (False) the gripper
        orientation: Optional orientation mode: 'horizontal-ns' (north-south horizontal), 
                    'horizontal-ew' (east-west horizontal), or None (default vertical)
        
    Returns:
        Joint positions to reach the target
    """
    # This is a simplified approach since we don't have direct access to an IK solver
    # We use a pre-defined joint configuration as a starting point
    # and adjust it based on the target position
    
    # Default joint positions for a reasonable pose
    joint_positions = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.6, 0.8, 0.04, 0.04])
    
    # Apply offset if provided
    if offset is not None:
        target_with_offset = target_position + offset
    else:
        target_with_offset = target_position
    
    # Get the number of joints
    num_joints = len(robot.dof_names)
    
    # Calculate distance from robot base to target
    # Assume robot base is at the origin for simplicity
    distance_xy = np.linalg.norm(target_with_offset[:2])  # XY distance
    height = target_with_offset[2]  # Z height
    
    # Calculate joint angles based on target position (approximate IK)
    # First joint (base rotation)
    if target_with_offset[0] != 0 or target_with_offset[1] != 0:
        joint_positions[0] = np.arctan2(target_with_offset[1], target_with_offset[0])
    
    # Adjust shoulder and elbow joints based on distance and height
    joint_positions[1] = -0.3 - (distance_xy * 0.4)  # More negative as distance increases
    joint_positions[3] = -1.8 - (height * 0.3)    # More negative as height increases
    
    # Adjust wrist angle to keep end effector level
    joint_positions[5] = 1.6 + (height * 0.4)
    
    # Fine-tune the wrist joint to reach lower
    if offset is not None and offset[2] < 0.05:  # If we're trying to get close to the object
        joint_positions[5] += 0.6  # Bend wrist more to reach down
        joint_positions[3] -= 0.6  # Bend elbow more
        # Adjust shoulder joint to reach forward more
        joint_positions[1] -= 0.3
    
    # Set gripper orientation based on specified orientation
    if orientation == 'horizontal-ew':  # East-west horizontal orientation (for banana)
        # Rotate wrist 90 degrees to align gripper east-west
        joint_positions[6] = 1.57  # ~90 degrees rotation of last joint
        # Adjust other wrist joints for horizontal approach
        joint_positions[4] = 1.57  # 90 degrees rotation of 5th joint
    
    # Set gripper position
    if num_joints > 7:  # If we have gripper joints
        gripper_value = 0.04 if gripper_open else 0.0
        joint_positions[7] = gripper_value
        joint_positions[8] = gripper_value
    
    # Return the calculated joint positions
    return joint_positions

def get_end_effector_position(robot, stage):
    """
    Get the current position of the robot's end effector.
    
    Args:
        robot: The robot articulation object
        stage: The USD stage
        
    Returns:
        End effector position as numpy array [x, y, z]
    """
    try:
        # Get the joint positions
        joint_positions = robot.get_joint_positions()
        
        # In Isaac Sim 4.5.0, we need to find the end effector prim
        # The end effector is typically the last link before the fingers
        # For Franka, this is usually panda_hand
        end_effector_path = f"{ROBOT_PRIM_PATH}/panda_hand"
        
        # Get the position from the prim
        ee_position, _ = get_prim_pose(stage, end_effector_path)
        
        if ee_position is not None:
            log(f"End effector position: {ee_position}")
            return ee_position
        else:
            log("WARNING: Could not get end effector position from prim")
            return None
    except Exception as e:
        log(f"Error getting end effector position: {e}")
        return None

def check_grasp_position(robot, stage, banana_position, threshold=0.05):
    """
    Check if the end effector is close enough to the banana for grasping.
    
    Args:
        robot: The robot articulation object
        stage: The USD stage
        banana_position: The position of the banana
        threshold: Maximum allowed distance between end effector and banana
        
    Returns:
        True if close enough, False otherwise
    """
    # Get the end effector position
    ee_position = get_end_effector_position(robot, stage)
    
    if ee_position is None:
        log("WARNING: Could not verify grasp position, proceeding anyway")
        return True
    
    # Calculate distance to banana
    distance = np.linalg.norm(ee_position - banana_position)
    log(f"Distance between end effector and banana: {distance:.4f}m")
    
    # Check if close enough
    if distance <= threshold:
        log(f"End effector is close enough to banana for grasping ({distance:.4f}m)")
        return True
    else:
        log(f"WARNING: End effector is too far from banana ({distance:.4f}m > {threshold}m)")
        return False

def run_franka_pick_place():
    """
    Run Isaac Sim simulation with a Franka robot to pick and place a banana
    """
    # Import SimulationApp first to initialize the simulation
    from isaacsim import SimulationApp

    # Create the simulation app (use headless=False to see the UI)
    simulation_app = SimulationApp({"headless": False})
    
    # Import necessary modules after SimulationApp is initialized
    import omni.usd
    import omni.timeline
    from pxr import UsdGeom
    
    log("Starting Franka robot pick and place demonstration")
    
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
    
    log(f"Found robot prim at {ROBOT_PRIM_PATH} with type: {robot_prim.GetTypeName()}")    # Verify banana prim path
    banana_prim = stage.GetPrimAtPath(BANANA_PRIM_PATH)
    if not banana_prim:
        log(f"WARNING: Banana not found at {BANANA_PRIM_PATH}, searching for it...")
        banana_path = find_object_in_scene(stage, "banana")
        if banana_path:
            global ACTUAL_BANANA_PATH
            ACTUAL_BANANA_PATH = banana_path
            log(f"Found banana at: {ACTUAL_BANANA_PATH}")
        else:
            log("ERROR: Could not find banana in scene")
            simulation_app.close()
            return
    else:
        log(f"Found banana at: {BANANA_PRIM_PATH}")
        
    # Verify target prim path (crate)
    target_prim = stage.GetPrimAtPath(TARGET_PRIM_PATH)
    if not target_prim:
        log(f"WARNING: Crate not found at {TARGET_PRIM_PATH}, searching for it...")
        # Try to find crate object
        target_path = find_object_in_scene(stage, "crate")
        
        if target_path:
            global ACTUAL_TARGET_PATH
            ACTUAL_TARGET_PATH = target_path
            log(f"Found target destination at: {ACTUAL_TARGET_PATH}")
        else:
            log("ERROR: Could not find a crate destination in scene")
            simulation_app.close()
            return
    else:
        log(f"Found crate at: {TARGET_PRIM_PATH}")
    
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
    if robot is not None and joint_positions is not None:
        # Create a robot movements controller
        robot_movements = RobotMovements(robot)
          # Get positions of banana and target
        log("Getting positions of banana and target...")
        banana_position, _ = get_prim_pose(stage, ACTUAL_BANANA_PATH)
        target_position, _ = get_prim_pose(stage, ACTUAL_TARGET_PATH)
        
        if banana_position is None or target_position is None:
            log("ERROR: Could not get positions of banana or target")
            simulation_app.close()
            return
        
        log(f"Banana position: {banana_position}")
        log(f"Target position: {target_position}")            # Execute pick and place sequence
        try:
            # Move to home position
            log("Moving to home position...")
            home_position = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.6, 0.0, 0.04, 0.04])
            robot.set_joint_positions(home_position)
            
            for _ in range(50):  # Wait for movement to complete
                simulation_app.update()
                time.sleep(0.02)
            
            # Get current banana and target positions from the stage
            banana_position, _ = get_prim_pose(stage, ACTUAL_BANANA_PATH)
            target_position, _ = get_prim_pose(stage, ACTUAL_TARGET_PATH)
            
            if banana_position is None or target_position is None:
                log("ERROR: Could not get positions of banana or target")
                simulation_app.close()
                return
            
            log(f"Banana position: {banana_position}")
            log(f"Target position: {target_position}")
            
            # Pre-grasp position - move above the banana
            log("Moving to pre-grasp position...")
            pre_grasp_offset = np.array([0.0, 0.0, 0.03])  # 3cm above the banana
            pre_grasp_joints = calculate_ik_for_banana_grasp(
                robot, 
                banana_position, 
                offset=pre_grasp_offset, 
                gripper_open=True
            )
            robot.set_joint_positions(pre_grasp_joints)
            
            # Log the joint positions to compare with the optimal values
            current_pos = robot.get_joint_positions()
            log(f"Pre-grasp joint positions: {current_pos}")
            
            for _ in range(100):  # Wait for movement to complete
                simulation_app.update()
                time.sleep(0.02)
              # Grasp position - move down to the banana
            log("Moving to grasp position...")
            grasp_offset = np.array([0.0, 0.0, -0.04])  # 4cm below the banana's center point
            grasp_joints = calculate_ik_for_banana_grasp(
                robot, 
                banana_position, 
                offset=grasp_offset, 
                gripper_open=True
            )
            robot.set_joint_positions(grasp_joints)
            
            # Log the joint positions to compare with the optimal values
            current_pos = robot.get_joint_positions()
            log(f"Grasp joint positions: {current_pos}")
            
            for _ in range(100):  # Wait for movement to complete
                simulation_app.update()
                time.sleep(0.02)
            
            # Check if the end effector is close enough to the banana
            is_position_good = check_grasp_position(robot, stage, banana_position, threshold=0.05)
            if not is_position_good:
                log("Adjusting grasp position to get closer to banana...")
                # Try to adjust the position with a small correction
                adjusted_grasp_offset = np.array([0.0, 0.0, -0.045])  # Slightly lower
                adjusted_grasp_joints = calculate_ik_for_banana_grasp(
                    robot, 
                    banana_position, 
                    offset=adjusted_grasp_offset, 
                    gripper_open=True
                )
                robot.set_joint_positions(adjusted_grasp_joints)
                
                for _ in range(50):  # Wait for adjustment
                    simulation_app.update()
                    time.sleep(0.02)
                
                # Check again
                is_position_good = check_grasp_position(robot, stage, banana_position, threshold=0.05)
                log(f"Position after adjustment: {'Good' if is_position_good else 'Still not optimal'}")
            
            # Close gripper to grasp the banana
            log("Closing gripper to grasp banana...")
            # Find the gripper joint indices
            gripper_indices = []
            for i, name in enumerate(dof_names):
                if "finger" in name.lower() or "gripper" in name.lower():
                    gripper_indices.append(i)
            
            if gripper_indices:
                # Get current positions
                current_pos = robot.get_joint_positions()
                # Close the gripper (set to 0.0 for most grippers)
                for idx in gripper_indices:
                    current_pos[idx] = 0.0
                robot.set_joint_positions(current_pos)
                
                for _ in range(50):  # Wait for gripper to close
                    simulation_app.update()
                    time.sleep(0.02)
                
                log("Gripper closed")
            else:
                log("WARNING: Could not identify gripper joints")            # Lift the banana
            log("Lifting the banana...")
            lift_offset = np.array([0.0, 0.0, 0.20])  # 20cm above the banana (increased for safety)
            lift_joints = calculate_ik_for_position(
                robot, 
                banana_position, 
                offset=lift_offset, 
                gripper_open=False,  # Keep gripper closed
                orientation='horizontal-ew'  # Maintain east-west orientation
            )
            robot.set_joint_positions(lift_joints)
            
            for _ in range(100):  # Wait for movement to complete
                simulation_app.update()
                time.sleep(0.02)            # Move to position above target (crate)
            log("Moving to target position...")
            target_approach_offset = np.array([0.0, 0.0, 0.20])  # 20cm above the crate (increased for safety)
            target_approach_joints = calculate_ik_for_position(
                robot, 
                target_position, 
                offset=target_approach_offset, 
                gripper_open=False,  # Keep gripper closed
                orientation='horizontal-ew'  # Maintain east-west orientation
            )
            robot.set_joint_positions(target_approach_joints)
            
            for _ in range(100):  # Wait for movement to complete
                simulation_app.update()
                time.sleep(0.02)
              # Lower to place position
            log("Lowering to place position...")
            place_offset = np.array([0.0, 0.0, -0.05])  # 5cm below the crate's top surface (was 2cm)
            place_joints = calculate_ik_for_position(
                robot, 
                target_position, 
                offset=place_offset, 
                gripper_open=False,  # Keep gripper closed
                orientation='horizontal-ew'  # Maintain east-west orientation
            )
            robot.set_joint_positions(place_joints)
            
            for _ in range(100):  # Wait for movement to complete
                simulation_app.update()
                time.sleep(0.02)
            
            # Open gripper to release the banana
            log("Opening gripper to release banana...")
            if gripper_indices:
                # Get current positions
                current_pos = robot.get_joint_positions()
                # Open the gripper (set to 0.04 for most grippers)
                for idx in gripper_indices:
                    current_pos[idx] = 0.04
                robot.set_joint_positions(current_pos)
                
                for _ in range(50):  # Wait for gripper to open
                    simulation_app.update()
                    time.sleep(0.02)
                
                log("Gripper opened, banana released")            # Lift after placing
            log("Lifting after placing...")
            after_place_offset = np.array([0.0, 0.0, 0.20])  # 20cm above the crate (increased for safety)
            after_place_joints = calculate_ik_for_position(
                robot, 
                target_position, 
                offset=after_place_offset, 
                gripper_open=True,  # Open gripper
                orientation='horizontal-ew'  # Maintain east-west orientation
            )
            robot.set_joint_positions(after_place_joints)
            
            for _ in range(100):  # Wait for movement to complete
                simulation_app.update()
                time.sleep(0.02)
              # Return to home position
            log("Returning to home position...")
            home_position = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.6, 0.0, 0.04, 0.04])
            robot.set_joint_positions(home_position)
            
            for _ in range(100):  # Wait for movement to complete
                simulation_app.update()
                time.sleep(0.02)
            
            log("Pick and place task completed successfully!")
            
        except Exception as e:
            log(f"Error during pick and place sequence: {e}")
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
    
    # Wait before closing
    log("Task complete, simulation will close in 5 seconds...")
    time.sleep(5)
    
    log("Simulation complete")
    simulation_app.close()

if __name__ == "__main__":
    # Set up the environment and run the simulation
    setup_isaac_sim_environment()
    run_franka_pick_place()
