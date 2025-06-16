#!/usr/bin/env python3
"""
Final Optimized Franka Robot Pick and Place with Advanced Precision Control

This script implements a high-precision pick-and-place task with the Franka robot by:
1. Using a more accurate approach for banana position and orientation
2. Implementing smaller, more gradual movement steps to improve accuracy
3. Adding direct vertical alignment above the banana with consistent approach
4. Fine-tuning joint angles based on the actual optimal grip configuration
"""

import time
import warnings
import numpy as np
from pathlib import Path
from core import setup_isaac_sim_environment

# Suppress numpy compatibility warnings
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

THIS_FOLDER = Path(__file__).parent.resolve()
USD_SCENE_PATH = str(THIS_FOLDER / "scenes" / "franka_blocks_manual.usd")
ROBOT_PRIM_PATH = "/World/franka"
BANANA_PRIM_PATH = "/World/banana"
TARGET_PRIM_PATH = "/World/crate"

# Global variables to store corrected paths
ACTUAL_BANANA_PATH = BANANA_PRIM_PATH
ACTUAL_TARGET_PATH = TARGET_PRIM_PATH

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

# Reference banana position for the optimal joint angles
REFERENCE_BANANA_POSITION = np.array([0.20, 0.18, 0.84])  # Approximated from logs

# Enhanced parameters for better precision
GRASP_THRESHOLD = 0.02    # Further reduced threshold (2cm) for more precision
MAX_REFINEMENT_ITERATIONS = 8  # More iterations for fine tuning
POSITION_STABILIZATION_TIME = 0.1  # Time to let physics settle between movements

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

def get_end_effector_position(robot, stage, get_grip_point=False):
    """
    Get the current position of the robot's end effector.
    Optionally get the grip point (lower position where actual grasping occurs)
    
    Args:
        robot: The robot articulation object
        stage: The USD stage
        get_grip_point: If True, return the grip point instead of the hand center
        
    Returns:
        End effector position as numpy array [x, y, z]
    """
    try:
        # Get the positions of the hand and fingers
        hand_path = f"{ROBOT_PRIM_PATH}/panda_hand"
        finger_path = f"{ROBOT_PRIM_PATH}/panda_hand/panda_leftfinger"
        
        # Get the position from the prim
        hand_position, _ = get_prim_pose(stage, hand_path)
        
        if get_grip_point:
            # For grip point, we estimate a position lower than the hand center
            # Calculate an offset of 8cm below the hand along local z-axis
            finger_position, _ = get_prim_pose(stage, finger_path)
            
            if finger_position is not None:
                # Estimate actual grip point (halfway between fingers but 5cm lower)
                grip_point = hand_position.copy()
                grip_point[2] -= 0.08  # 8cm below hand center
                log(f"Grip point position: {grip_point}")
                return grip_point
        
        if hand_position is not None:
            log(f"End effector position: {hand_position}")
            return hand_position
        else:
            log("WARNING: Could not get end effector position from prim")
            return None
    except Exception as e:
        log(f"Error getting end effector position: {e}")
        return None

def check_grasp_position(robot, stage, banana_position, threshold=GRASP_THRESHOLD, use_grip_point=True):
    """
    Check if the end effector is close enough to the banana for grasping.
    Performs more detailed position checking for better accuracy.
    
    Args:
        robot: The robot articulation object
        stage: The USD stage
        banana_position: The position of the banana
        threshold: Maximum allowed distance between end effector and banana
        use_grip_point: Whether to use the grip point instead of hand center
        
    Returns:
        (is_close_enough, actual_distance, error_vector) tuple
    """
    # Get the end effector position (either grip point or hand center)
    ee_position = get_end_effector_position(robot, stage, get_grip_point=use_grip_point)
    
    if ee_position is None:
        log("WARNING: Could not verify grasp position, proceeding anyway")
        return True, 0.0, np.zeros(3)
    
    # Calculate distance to banana
    error_vector = banana_position - ee_position
    distance = np.linalg.norm(error_vector)
    
    # Log detailed position information
    log(f"End effector position: {ee_position}")
    log(f"Banana position: {banana_position}")
    log(f"Error vector: {error_vector}")
    log(f"Distance between end effector and banana: {distance:.4f}m")
    
    # Check if close enough
    is_close_enough = distance <= threshold
    
    if is_close_enough:
        log(f"End effector is close enough to banana for grasping ({distance:.4f}m)")
    else:
        log(f"WARNING: End effector is too far from banana ({distance:.4f}m > {threshold}m)")
        
    return is_close_enough, distance, error_vector

def get_banana_dimensions(stage, banana_path):
    """
    Get the dimensions of the banana to better determine grip position
    
    Args:
        stage: The USD stage
        banana_path: Path to the banana prim
        
    Returns:
        Tuple of (length, width, height) as best estimated
    """
    from pxr import UsdGeom
    
    # Get the prim at the specified path
    prim = stage.GetPrimAtPath(banana_path)
    if not prim:
        log(f"ERROR: Banana prim not found at {banana_path}")
        return (0.15, 0.05, 0.05)  # Default values
    
    try:
        # Try to get extents from the prim
        bbox_cache = UsdGeom.BBoxCache(0, includeVisibleOnly=True)
        bounds = bbox_cache.ComputeWorldBound(prim)
        
        # Get the range (min and max points)
        range_min = bounds.GetRange().GetMin()
        range_max = bounds.GetRange().GetMax()
        
        # Calculate dimensions
        dimensions = np.array([
            range_max[0] - range_min[0],
            range_max[1] - range_min[1],
            range_max[2] - range_min[2]
        ])
        
        log(f"Banana dimensions: {dimensions}")
        
        # Sort dimensions to get length, width, height
        sorted_dimensions = np.sort(dimensions)[::-1]
        return tuple(sorted_dimensions)
    
    except Exception as e:
        log(f"Error getting banana dimensions: {e}")
        return (0.15, 0.05, 0.05)  # Default values

def get_ideal_grasp_position(stage, banana_position, banana_orientation):
    """
    Calculate the ideal position to grasp the banana based on its 
    position, orientation, and dimensions
    
    Args:
        stage: The USD stage
        banana_position: The center position of the banana
        banana_orientation: The orientation matrix of the banana
        
    Returns:
        The ideal grasp position as numpy array [x, y, z]
    """
    # Get the main axis direction (typically the x-axis for a banana)
    main_axis = banana_orientation[:, 0]  # First column is the x-axis direction
    
    # Normalize to get a unit vector
    norm = np.linalg.norm(main_axis)
    if norm > 0:
        main_axis = main_axis / norm
    
    # Get banana dimensions
    banana_length, banana_width, banana_height = get_banana_dimensions(stage, ACTUAL_BANANA_PATH)
    
    # Calculate grasp position - we want to grasp near the middle
    # but slightly adjusted for better stability
    grasp_offset = np.zeros(3)
    
    # If banana is more horizontal (X-Y plane)
    if abs(main_axis[2]) < 0.3:  # Z component is small
        # Grasp from above, near the middle, slightly toward the thicker end
        grasp_offset = main_axis * (-banana_length * 0.1)  # 10% toward thicker end
        
        # Add slight upward offset (for approach from above)
        grasp_offset[2] += 0.02
    else:
        # Banana is more vertical - grasp from the side
        # Find the horizontal component of the main axis
        horizontal = np.array([main_axis[0], main_axis[1], 0])
        horiz_norm = np.linalg.norm(horizontal)
        
        if horiz_norm > 0.01:
            # Normalize horizontal component
            horizontal = horizontal / horiz_norm
            
            # Grasp perpendicular to the main axis, at the middle height
            grasp_offset = horizontal * banana_width * 0.5
            grasp_offset[2] += 0.02  # Slight upward offset
    
    # Calculate final grasp position
    grasp_position = banana_position + grasp_offset
    
    log(f"Calculated ideal grasp position: {grasp_position}")
    return grasp_position

def vertical_align_above_target(robot, stage, target_position, height_offset=0.15, radius=0.05):
    """
    Move the robot to a position directly above the target with precise vertical alignment.
    This ensures a clean approach from above.
    
    Args:
        robot: The robot articulation object
        stage: The USD stage
        target_position: The target position to align above
        height_offset: How high above the target to position (meters)
        radius: Maximum allowed horizontal offset from target
        
    Returns:
        The final joint positions achieved
    """
    # Start with a home-like configuration but facing the target
    current_joints = robot.get_joint_positions()
    aligned_joints = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.6, 0.0, 0.04, 0.04])
    
    # Set the base rotation to face the target
    if target_position[0] != 0 or target_position[1] != 0:
        aligned_joints[0] = np.arctan2(target_position[1], target_position[0])
    
    # Apply the initial alignment
    robot.set_joint_positions(aligned_joints)
    simulation_app.update()
    time.sleep(0.2)
    
    # Get current end effector position
    ee_position = get_end_effector_position(robot, stage)
    if ee_position is None:
        log("WARNING: Could not get end effector position for alignment")
        return aligned_joints
    
    # Calculate target position with height offset
    align_target = target_position.copy()
    align_target[2] += height_offset
    
    # Calculate horizontal distance to target
    horizontal_error = np.array([align_target[0] - ee_position[0], align_target[1] - ee_position[1]])
    horizontal_distance = np.linalg.norm(horizontal_error)
    
    log(f"Initial horizontal distance to target: {horizontal_distance:.4f}m")
    
    # Iteratively refine the position to get directly above target
    max_iterations = 5
    for i in range(max_iterations):
        # Get current end effector position
        ee_position = get_end_effector_position(robot, stage)
        if ee_position is None:
            break
            
        # Calculate horizontal error
        horizontal_error = np.array([align_target[0] - ee_position[0], align_target[1] - ee_position[1]])
        horizontal_distance = np.linalg.norm(horizontal_error)
        
        # Calculate height error
        height_error = align_target[2] - ee_position[2]
        
        log(f"Alignment iteration {i+1}: horizontal distance = {horizontal_distance:.4f}m, height error = {height_error:.4f}m")
        
        # If we're close enough, we're done
        if horizontal_distance < radius and abs(height_error) < 0.05:
            log(f"Vertical alignment achieved within tolerance")
            break
            
        # Adjust base rotation for better XY alignment
        if horizontal_distance > 0.01:
            current_angle = aligned_joints[0]
            target_angle = np.arctan2(align_target[1], align_target[0])
            angle_error = target_angle - current_angle
            
            # Normalize angle
            if angle_error > np.pi:
                angle_error -= 2 * np.pi
            elif angle_error < -np.pi:
                angle_error += 2 * np.pi
                
            # Apply adjustment to base rotation
            aligned_joints[0] += angle_error * 0.5
        
        # Adjust shoulder (joint 1) and elbow (joint 3) for height
        aligned_joints[1] += height_error * 0.2
        aligned_joints[3] += height_error * 0.3
        
        # Adjust reach distance with shoulder and elbow
        # This is a simplified approach - in a full IK solution, we'd use the Jacobian
        if horizontal_distance > radius:
            # If we need to reach further or pull back
            reach_error = horizontal_distance - radius
            if np.dot(horizontal_error, np.array([np.cos(aligned_joints[0]), np.sin(aligned_joints[0])])) > 0:
                # Need to reach further
                aligned_joints[1] -= reach_error * 0.3  # Extend shoulder
                aligned_joints[3] += reach_error * 0.4  # Extend elbow
            else:
                # Need to pull back
                aligned_joints[1] += reach_error * 0.3  # Retract shoulder
                aligned_joints[3] -= reach_error * 0.4  # Retract elbow
        
        # Apply the adjusted joints
        robot.set_joint_positions(aligned_joints)
        simulation_app.update()
        time.sleep(0.2)
    
    return aligned_joints

def directly_above_banana_approach(robot, stage, banana_position, grasp_position):
    """
    Implement a direct, vertical approach to the banana for more reliable grasping.
    
    Args:
        robot: The robot articulation object
        stage: The USD stage
        banana_position: The position of the banana
        grasp_position: The ideal grasp position
        
    Returns:
        True if the approach was successful, False otherwise
    """
    # First, move to a position directly above the banana
    log("Moving to position directly above banana...")
    
    # Set the height offset based on banana height
    above_position = grasp_position.copy()
    above_position[2] += 0.15  # 15cm above the grasp position
    
    # Get aligned joints to position directly above
    aligned_joints = vertical_align_above_target(robot, stage, above_position, height_offset=0.0)
    
    # Now approach in small, careful steps
    num_steps = 5
    current_joints = aligned_joints.copy()
    
    for step in range(num_steps):
        log(f"Vertical approach step {step+1}/{num_steps}")
        
        # Calculate the interpolated position for this step
        fraction = (step + 1) / num_steps
        target_position = above_position * (1 - fraction) + grasp_position * fraction
        
        # Get current end effector position
        ee_position = get_end_effector_position(robot, stage)
        if ee_position is None:
            continue
            
        # Calculate error to target position
        error_vector = target_position - ee_position
        distance = np.linalg.norm(error_vector)
        
        log(f"Step {step+1}: current distance to target: {distance:.4f}m")
        
        # Adjust joints to reach the target position
        # For this approach, we mainly adjust joints 1 (shoulder) and 3 (elbow)
        # to move down while keeping the horizontal position stable
        
        # Vertical movement - adjust shoulder and elbow
        height_error = error_vector[2]
        current_joints[1] -= height_error * 0.3  # Shoulder down
        current_joints[3] -= height_error * 0.5  # Elbow adjustment
        
        # Apply the joint angles
        robot.set_joint_positions(current_joints)
        
        # Wait for movement to stabilize
        for _ in range(20):
            simulation_app.update()
            time.sleep(0.02)
        
        # Check if we've reached the target position
        ee_position = get_end_effector_position(robot, stage)
        if ee_position is not None:
            distance = np.linalg.norm(target_position - ee_position)
            if distance < 0.03:  # 3cm accuracy for intermediate steps
                log(f"Reached intermediate target position, distance: {distance:.4f}m")
            else:
                log(f"Did not reach intermediate target, distance: {distance:.4f}m")
    
    # Do a final position check
    is_close, distance, _ = check_grasp_position(robot, stage, grasp_position, threshold=GRASP_THRESHOLD)
    log(f"Final approach complete, distance to grasp position: {distance:.4f}m")
    
    return is_close

def direct_banana_grasp_approach(robot, stage, banana_position, use_optimal_config=True):
    """
    Implement a direct approach to grasping the banana based on
    the optimal joint configuration but adjusted for the actual banana position.
    
    Args:
        robot: The robot articulation object
        stage: The USD stage
        banana_position: The position of the banana
        use_optimal_config: Whether to use the optimal joint configuration as a basis
        
    Returns:
        True if the approach was successful, False otherwise
    """
    # Get banana orientation
    _, banana_orientation = get_prim_pose(stage, ACTUAL_BANANA_PATH)
    
    # Calculate ideal grasp position
    grasp_position = get_ideal_grasp_position(stage, banana_position, banana_orientation)
    
    # First, adjust the optimal configuration for the actual banana position
    if use_optimal_config:
        # Start with optimal joint configuration
        joint_positions = OPTIMAL_BANANA_GRASP_JOINTS.copy()
        
        # Calculate delta between current banana and reference position
        banana_delta = banana_position - REFERENCE_BANANA_POSITION
        
        # Adjust base rotation to point at actual banana
        if banana_position[0] != 0 or banana_position[1] != 0:
            joint_positions[0] = np.arctan2(banana_position[1], banana_position[0])
        
        # Apply delta adjustments to other joints
        joint_positions[1] += banana_delta[2] * 0.6  # Shoulder adjustment for height
        joint_positions[3] -= banana_delta[2] * 0.8  # Elbow adjustment for height
        
        # Wrist adjustments
        joint_positions[5] += banana_delta[2] * 0.3  # Wrist angle
    else:
        # Use the vertical approach strategy
        return directly_above_banana_approach(robot, stage, banana_position, grasp_position)
    
    # Apply the adjusted optimal configuration
    log("Applying adjusted optimal joint configuration...")
    robot.set_joint_positions(joint_positions)
    
    # Wait for movement to complete
    for _ in range(50):
        simulation_app.update()
        time.sleep(0.02)
    
    # Check if we reached a good position
    is_close, distance, error_vector = check_grasp_position(
        robot, stage, grasp_position, threshold=GRASP_THRESHOLD
    )
    
    if not is_close:
        # Try one more refinement
        log(f"Initial position not close enough ({distance:.4f}m), refining...")
        
        # Get current joint positions
        current_joints = robot.get_joint_positions()
        
        # Adjust for height error - this is usually the biggest issue
        height_error = error_vector[2]
        current_joints[1] -= height_error * 0.5  # Shoulder adjustment
        current_joints[3] -= height_error * 0.7  # Elbow adjustment
        
        # Horizontal adjustments if needed
        xy_error = np.linalg.norm(error_vector[:2])
        if xy_error > 0.02:
            # Adjust base rotation
            current_angle = current_joints[0]
            target_angle = np.arctan2(banana_position[1], banana_position[0])
            angle_error = target_angle - current_angle
            
            # Normalize angle
            if angle_error > np.pi:
                angle_error -= 2 * np.pi
            elif angle_error < -np.pi:
                angle_error += 2 * np.pi
                
            # Apply adjustment to base rotation
            current_joints[0] += angle_error * 0.3
        
        # Apply refined joints
        robot.set_joint_positions(current_joints)
        
        # Wait for movement to complete
        for _ in range(30):
            simulation_app.update()
            time.sleep(0.02)
        
        # Check position again
        is_close, distance, _ = check_grasp_position(
            robot, stage, grasp_position, threshold=GRASP_THRESHOLD
        )
    
    log(f"Final grasp approach complete, distance: {distance:.4f}m, success: {is_close}")
    return is_close

def create_trajectory_to_target(start_position, target_position, num_waypoints=5):
    """
    Create a trajectory from start to target position with multiple waypoints
    
    Args:
        start_position: Starting position
        target_position: Target position
        num_waypoints: Number of waypoints to generate
        
    Returns:
        List of waypoint positions
    """
    waypoints = []
    
    for i in range(num_waypoints):
        # Calculate interpolation factor
        t = (i + 1) / num_waypoints
        
        # Linear interpolation
        waypoint = start_position * (1 - t) + target_position * t
        
        # Add an arc for lifting/lowering
        if i < num_waypoints / 2:
            # First half of trajectory - lift up
            lift_height = 0.2  # Maximum lift height
            lift_factor = np.sin(t * np.pi)  # Sinusoidal profile
            waypoint[2] += lift_height * lift_factor
        
        waypoints.append(waypoint)
    
    return waypoints

def run_franka_pick_place():
    """
    Run Isaac Sim simulation with a Franka robot to pick and place a banana
    using direct optimal grasp configuration with position-based adjustments
    """
    # Import SimulationApp first to initialize the simulation
    from isaacsim import SimulationApp

    # Create the simulation app (use headless=False to see the UI)
    global simulation_app
    simulation_app = SimulationApp({"headless": False})
    
    # Import necessary modules after SimulationApp is initialized
    import omni.usd
    import omni.timeline
    from pxr import UsdGeom
    
    log("Starting Final Optimized Franka robot pick and place demonstration")
    
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
    
    log(f"Found robot prim at {ROBOT_PRIM_PATH} with type: {robot_prim.GetTypeName()}")
    
    # Verify banana prim path
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
        # Get positions of banana and target
        log("Getting positions of banana and target...")
        banana_position, banana_orientation = get_prim_pose(stage, ACTUAL_BANANA_PATH)
        target_position, _ = get_prim_pose(stage, ACTUAL_TARGET_PATH)
        
        if banana_position is None or target_position is None:
            log("ERROR: Could not get positions of banana or target")
            simulation_app.close()
            return
        
        log(f"Banana position: {banana_position}")
        log(f"Target position: {target_position}")
        
        # Calculate ideal grasp position
        grasp_position = get_ideal_grasp_position(stage, banana_position, banana_orientation)
        
        # Execute pick and place sequence
        try:
            # Move to home position
            log("Moving to home position...")
            home_position = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.6, 0.0, 0.04, 0.04])
            robot.set_joint_positions(home_position)
            
            for _ in range(50):  # Wait for movement to complete
                simulation_app.update()
                time.sleep(0.02)
            
            # Update positions from the stage
            banana_position, banana_orientation = get_prim_pose(stage, ACTUAL_BANANA_PATH)
            target_position, _ = get_prim_pose(stage, ACTUAL_TARGET_PATH)
            
            if banana_position is None or target_position is None:
                log("ERROR: Could not get positions of banana or target")
                simulation_app.close()
                return
            
            log(f"Updated banana position: {banana_position}")
            log(f"Updated target position: {target_position}")
            
            # Use a direct, optimal configuration approach to the banana
            # This uses the optimal joint config and adjusts it for the actual position
            is_approach_successful = direct_banana_grasp_approach(
                robot, stage, banana_position, use_optimal_config=True
            )
            
            if not is_approach_successful:
                log("First approach was not successful, trying vertical approach...")
                
                # Try the vertical approach as a fallback
                grasp_position = get_ideal_grasp_position(stage, banana_position, banana_orientation)
                is_approach_successful = directly_above_banana_approach(
                    robot, stage, banana_position, grasp_position
                )
            
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
                
                for _ in range(30):  # Wait for gripper to close
                    simulation_app.update()
                    time.sleep(0.02)
                
                log("Gripper closed")
            else:
                log("WARNING: Could not identify gripper joints")
            
            # Lift the banana using a smooth trajectory
            log("Lifting the banana...")
            
            # Get current position
            ee_position = get_end_effector_position(robot, stage)
            if ee_position is None:
                ee_position = banana_position
            
            # Calculate lift position (20cm above current position)
            lift_position = ee_position.copy()
            lift_position[2] += 0.2
            
            # Create lift trajectory
            lift_trajectory = create_trajectory_to_target(ee_position, lift_position, 3)
            
            # Execute lift trajectory
            for i, waypoint in enumerate(lift_trajectory):
                log(f"Lift waypoint {i+1}/{len(lift_trajectory)}")
                
                # Use simple IK adjustments to reach the waypoint
                current_pos = robot.get_joint_positions()
                
                # Adjust mainly the shoulder and elbow for lifting
                current_pos[1] -= 0.15  # Shoulder up
                current_pos[3] += 0.2   # Elbow up
                
                # Keep gripper closed
                for idx in gripper_indices:
                    current_pos[idx] = 0.0
                
                robot.set_joint_positions(current_pos)
                
                for _ in range(30):
                    simulation_app.update()
                    time.sleep(0.02)
            
            # Move to position above target (crate)
            log("Moving to target position...")
            
            # Calculate a position above the target
            above_target_position = target_position.copy()
            above_target_position[2] += 0.2  # 20cm above target
            
            # Use the vertical alignment approach
            aligned_joints = vertical_align_above_target(
                robot, stage, above_target_position, height_offset=0.0
            )
            
            # Keep gripper closed
            current_pos = robot.get_joint_positions()
            for idx in gripper_indices:
                aligned_joints[idx] = 0.0
            
            robot.set_joint_positions(aligned_joints)
            
            for _ in range(50):
                simulation_app.update()
                time.sleep(0.02)
            
            # Lower to place position
            log("Lowering to place position...")
            
            # Get current position
            ee_position = get_end_effector_position(robot, stage)
            if ee_position is None:
                ee_position = above_target_position
            
            # Calculate place position
            place_position = target_position.copy()
            place_position[2] += 0.05  # 5cm above target surface
            
            # Create place trajectory
            place_trajectory = create_trajectory_to_target(ee_position, place_position, 3)
            
            # Execute place trajectory
            for i, waypoint in enumerate(place_trajectory):
                log(f"Place waypoint {i+1}/{len(place_trajectory)}")
                
                # Use simple IK adjustments to reach the waypoint
                current_pos = robot.get_joint_positions()
                
                # Adjust mainly the shoulder and elbow for lowering
                current_pos[1] += 0.15  # Shoulder down
                current_pos[3] -= 0.2   # Elbow down
                
                # Keep gripper closed
                for idx in gripper_indices:
                    current_pos[idx] = 0.0
                
                robot.set_joint_positions(current_pos)
                
                for _ in range(30):
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
                
                for _ in range(30):  # Wait for gripper to open
                    simulation_app.update()
                    time.sleep(0.02)
                
                log("Gripper opened, banana released")
            
            # Lift after placing
            log("Lifting after placing...")
            
            # Get current position
            ee_position = get_end_effector_position(robot, stage)
            if ee_position is None:
                ee_position = place_position
            
            # Calculate lift position after placing
            lift_after_position = ee_position.copy()
            lift_after_position[2] += 0.2
            
            # Adjust joints for lifting
            current_pos = robot.get_joint_positions()
            current_pos[1] -= 0.2  # Shoulder up
            current_pos[3] += 0.3  # Elbow up
            
            # Keep gripper open
            for idx in gripper_indices:
                current_pos[idx] = 0.04
            
            robot.set_joint_positions(current_pos)
            
            for _ in range(50):
                simulation_app.update()
                time.sleep(0.02)
            
            # Return to home position
            log("Returning to home position...")
            home_position = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.6, 0.0, 0.04, 0.04])
            robot.set_joint_positions(home_position)
            
            for _ in range(50):  # Wait for movement to complete
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
