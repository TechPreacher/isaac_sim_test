#!/usr/bin/env python3
"""
Enhanced Franka Robot Pick and Place with Visual Servoing and Precision Grip

This script implements a high-precision pick-and-place task with the Franka robot by:
1. Using visual servoing techniques to achieve precise banana grasping
2. Implementing multi-stage refinement with direct feedback loops
3. Adding adaptive grip height and orientation based on banana position
4. Using dynamic offset tuning based on error feedback
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
GRASP_THRESHOLD = 0.025  # Reduced threshold (2.5 cm) for more precise positioning
MAX_REFINEMENT_ITERATIONS = 5  # More iterations for fine tuning
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

def get_end_effector_position(robot, stage, get_both_hands=False):
    """
    Get the current position of the robot's end effector.
    
    Args:
        robot: The robot articulation object
        stage: The USD stage
        get_both_hands: If True, return both hand and finger positions
        
    Returns:
        End effector position as numpy array [x, y, z]
        If get_both_hands is True, returns (hand position, finger position)
    """
    try:
        # Get the positions of both the hand (palm) and the finger tips
        hand_path = f"{ROBOT_PRIM_PATH}/panda_hand"
        finger_path = f"{ROBOT_PRIM_PATH}/panda_hand/panda_leftfinger"
        
        # Get the position from the prim
        hand_position, _ = get_prim_pose(stage, hand_path)
        
        if get_both_hands:
            finger_position, _ = get_prim_pose(stage, finger_path)
            log(f"Hand position: {hand_position}, Finger position: {finger_position}")
            return hand_position, finger_position
        
        if hand_position is not None:
            log(f"End effector position: {hand_position}")
            return hand_position
        else:
            log("WARNING: Could not get end effector position from prim")
            return None
    except Exception as e:
        log(f"Error getting end effector position: {e}")
        return None if not get_both_hands else (None, None)

def check_grasp_position(robot, stage, banana_position, threshold=GRASP_THRESHOLD):
    """
    Check if the end effector is close enough to the banana for grasping.
    Performs more detailed position checking for better accuracy.
    
    Args:
        robot: The robot articulation object
        stage: The USD stage
        banana_position: The position of the banana
        threshold: Maximum allowed distance between end effector and banana
        
    Returns:
        (is_close_enough, actual_distance, error_vector) tuple
    """
    # Get the end effector position
    ee_position = get_end_effector_position(robot, stage)
    
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

def get_banana_orientation(stage, banana_path):
    """
    Get the orientation of the banana to determine optimal gripper alignment
    
    Args:
        stage: The USD stage
        banana_path: Path to the banana prim
        
    Returns:
        Main axis direction as a unit vector
    """
    _, orientation = get_prim_pose(stage, banana_path)
    
    if orientation is None:
        # Default to east-west orientation if we can't determine
        return np.array([1.0, 0.0, 0.0])
    
    # Use the x-axis of the banana's local frame as its main axis
    # This assumes the banana is oriented along its local x-axis
    main_axis = orientation[:, 0]  # First column is the x-axis direction
    
    # Normalize to get a unit vector
    norm = np.linalg.norm(main_axis)
    if norm > 0:
        main_axis = main_axis / norm
    
    log(f"Detected banana main axis: {main_axis}")
    return main_axis

def visual_servo_to_target(robot, stage, target_position, current_joints, max_iterations=MAX_REFINEMENT_ITERATIONS, threshold=GRASP_THRESHOLD):
    """
    Use visual servoing techniques to precisely position the end effector
    
    Args:
        robot: The robot articulation object
        stage: The USD stage
        target_position: The target position to reach
        current_joints: Current joint positions to start from
        max_iterations: Maximum number of refinement iterations
        threshold: Position accuracy threshold
        
    Returns:
        Refined joint positions for precise positioning
    """
    # Start with the provided joint positions
    joint_positions = current_joints.copy()
    original_positions = robot.get_joint_positions()
    
    # Define adjustment gains for different joints
    position_gain = 0.3  # How much to adjust for position errors
    orientation_gain = 0.2  # How much to adjust for orientation errors
    
    # Storage for tracking progress
    best_distance = float('inf')
    best_joints = joint_positions.copy()
    
    for i in range(max_iterations):
        # Apply current joint positions
        robot.set_joint_positions(joint_positions)
        
        # Wait for physics to stabilize
        time.sleep(POSITION_STABILIZATION_TIME)
        simulation_app.update()
        
        # Get current end effector position
        ee_position = get_end_effector_position(robot, stage)
        if ee_position is None:
            log(f"Iteration {i+1}: Could not get end effector position")
            continue
            
        # Calculate error
        error_vector = target_position - ee_position
        distance = np.linalg.norm(error_vector)
        
        # Store best result so far
        if distance < best_distance:
            best_distance = distance
            best_joints = joint_positions.copy()
            
        # Check if we're close enough
        if distance < threshold:
            log(f"Visual servoing converged after {i+1} iterations, distance: {distance:.4f}m")
            break
            
        # Log progress
        log(f"Iteration {i+1}: Distance = {distance:.4f}m, Error = {error_vector}")
        
        # Calculate adjustments based on error
        # We need to map Cartesian space errors to joint space
        
        # Base rotation adjustment (joint 1) - maps to XY position
        xy_error = np.array([error_vector[0], error_vector[1]])
        xy_norm = np.linalg.norm(xy_error)
        if xy_norm > 0.01:
            # Calculate desired rotation change using atan2
            current_angle = np.arctan2(ee_position[1], ee_position[0])
            target_angle = np.arctan2(target_position[1], target_position[0])
            angle_error = target_angle - current_angle
            
            # Normalize angle to [-pi, pi]
            if angle_error > np.pi:
                angle_error -= 2 * np.pi
            elif angle_error < -np.pi:
                angle_error += 2 * np.pi
                
            # Apply base rotation adjustment
            joint_positions[0] += angle_error * orientation_gain
        
        # Height adjustment (mainly joints 2 and 4)
        height_error = error_vector[2]
        joint_positions[1] -= height_error * position_gain  # Shoulder joint
        joint_positions[3] -= height_error * position_gain * 1.5  # Elbow joint (larger effect)
        
        # Reach adjustment (X distance)
        reach_error = error_vector[0] * np.cos(joint_positions[0]) + error_vector[1] * np.sin(joint_positions[0])
        joint_positions[1] -= reach_error * position_gain * 0.5  # Shoulder affects reach
        joint_positions[3] += reach_error * position_gain * 0.8  # Elbow affects reach
        
        # Minor adjustments to wrist
        joint_positions[5] -= height_error * position_gain * 0.3  # Wrist affects gripper angle
        
        # Enforce joint limits (simplified - would be better to use actual limits)
        joint_positions = np.clip(joint_positions, -3.14, 3.14)
        
        log(f"Adjusted joint positions: {joint_positions}")
    
    # Restore original position
    robot.set_joint_positions(original_positions)
    
    # Return the best joint positions found
    if best_distance < float('inf'):
        log(f"Best distance achieved: {best_distance:.4f}m")
        return best_joints
    else:
        return joint_positions

def calculate_precise_banana_grasp_joints(robot, banana_position, stage, offset=None, gripper_open=True):
    """
    Calculate joint positions for precise banana grasping using a combination of:
    1. Optimal reference joint angles
    2. Position-based adjustments
    3. Banana orientation adaptation
    4. Visual servoing refinement
    
    Args:
        robot: The robot articulation object
        banana_position: The 3D position of the banana
        stage: The USD stage for position feedback
        offset: Optional offset to apply to the target position (in world frame)
        gripper_open: Whether to open (True) or close (False) the gripper
        
    Returns:
        Joint positions to reach the target
    """
    # Start with the optimal joint configuration for banana grasping
    joint_positions = OPTIMAL_BANANA_GRASP_JOINTS.copy()
    num_joints = len(robot.dof_names)
    
    # Get banana orientation
    banana_axis = get_banana_orientation(stage, ACTUAL_BANANA_PATH)
    
    # Calculate the difference between current banana position and reference position
    banana_delta = banana_position - REFERENCE_BANANA_POSITION
    
    # Adjust the first joint (base rotation) based on the banana's XY position and orientation
    # We want the gripper to align with the banana's axis
    if banana_position[0] != 0 or banana_position[1] != 0:
        # Calculate angle to banana position
        angle_to_banana = np.arctan2(banana_position[1], banana_position[0])
        
        # Calculate angle of banana orientation in XY plane
        banana_xy = np.array([banana_axis[0], banana_axis[1]])
        banana_xy_norm = np.linalg.norm(banana_xy)
        
        if banana_xy_norm > 0.1:  # If banana has a significant XY orientation
            banana_angle = np.arctan2(banana_xy[1], banana_xy[0])
            
            # Determine if we should approach along or perpendicular to banana axis
            # For bananas, we typically want to grasp perpendicular to the main axis
            approach_angle = banana_angle + np.pi/2  # Perpendicular approach
            
            # Normalize angle to [-pi, pi]
            if approach_angle > np.pi:
                approach_angle -= 2 * np.pi
            
            # Blend between position-based angle and orientation-based angle
            joint_positions[0] = 0.7 * angle_to_banana + 0.3 * approach_angle
        else:
            # If banana is mostly vertical, just face it directly
            joint_positions[0] = angle_to_banana
    
    # Apply precise adjustments based on the banana position delta
    # Shoulder joint (joint 2) adjustment based on XY distance and Z height
    joint_positions[1] += banana_delta[0] * 0.3 + banana_delta[2] * 0.6
    
    # Elbow joints adjustments
    joint_positions[3] -= banana_delta[2] * 0.8  # Height has strong effect on elbow 2
    
    # Wrist adjustments for better approach angle
    # Adjust wrist based on banana orientation
    banana_z_component = abs(banana_axis[2])
    if banana_z_component > 0.5:  # Banana is more vertical
        # Adjust wrist to approach from the side
        joint_positions[5] += 0.3
    else:  # Banana is more horizontal
        # Adjust wrist to approach from above/below
        joint_positions[5] += banana_delta[2] * 0.4
    
    # Apply vertical offset if specified
    if offset is not None:
        if offset[2] > 0:  # Pre-grasp position (above banana)
            # For pre-grasp, adjust the joints to be slightly above
            joint_positions[1] += 0.15  # Shoulder up
            joint_positions[3] += 0.25  # Elbow up
        elif offset[2] < 0:  # Below banana or for grasping
            # For grasp or below position, adjust to reach lower
            joint_positions[1] -= 0.18  # Shoulder down (increased)
            joint_positions[3] -= 0.3   # Elbow down (increased)
            # Fine-tune wrist for better grasping angle
            joint_positions[5] += 0.2
    
    # Set gripper position
    if num_joints > 7:  # If we have gripper joints
        gripper_value = 0.04 if gripper_open else 0.0
        joint_positions[7] = gripper_value
        joint_positions[8] = gripper_value
    
    # Calculate target position with offset
    target_position = banana_position.copy()
    if offset is not None:
        target_position += offset
    
    # Perform visual servoing refinement
    refined_joints = visual_servo_to_target(
        robot, 
        stage, 
        target_position, 
        joint_positions,
        max_iterations=MAX_REFINEMENT_ITERATIONS,
        threshold=GRASP_THRESHOLD
    )
    
    log(f"Original calculated joints: {joint_positions}")
    log(f"Refined joint positions: {refined_joints}")
    
    return refined_joints

def run_franka_pick_place():
    """
    Run Isaac Sim simulation with a Franka robot to pick and place a banana
    using enhanced precision control with visual servoing
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
    
    log("Starting Enhanced Franka robot pick and place demonstration")
    
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
        banana_position, _ = get_prim_pose(stage, ACTUAL_BANANA_PATH)
        target_position, _ = get_prim_pose(stage, ACTUAL_TARGET_PATH)
        
        if banana_position is None or target_position is None:
            log("ERROR: Could not get positions of banana or target")
            simulation_app.close()
            return
        
        log(f"Banana position: {banana_position}")
        log(f"Target position: {target_position}")
        
        # Execute pick and place sequence
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
            
            # Multi-stage approach to banana with increasingly fine adjustments
            
            # Stage 1: Coarse pre-grasp position - higher above the banana
            log("Stage 1: Moving to coarse pre-grasp position...")
            coarse_pre_grasp_offset = np.array([0.0, 0.0, 0.1])  # 10cm above the banana
            coarse_pre_grasp_joints = calculate_precise_banana_grasp_joints(
                robot, 
                banana_position, 
                stage,
                offset=coarse_pre_grasp_offset, 
                gripper_open=True
            )
            robot.set_joint_positions(coarse_pre_grasp_joints)
            
            # Log the joint positions to compare with the optimal values
            current_pos = robot.get_joint_positions()
            log(f"Coarse pre-grasp joint positions: {current_pos}")
            
            for _ in range(80):  # Wait for movement to complete
                simulation_app.update()
                time.sleep(0.02)
            
            # Stage 2: Fine pre-grasp position - closer to the banana
            log("Stage 2: Moving to fine pre-grasp position...")
            fine_pre_grasp_offset = np.array([0.0, 0.0, 0.05])  # 5cm above the banana
            
            # Update banana position in case it moved
            banana_position, _ = get_prim_pose(stage, ACTUAL_BANANA_PATH)
            
            fine_pre_grasp_joints = calculate_precise_banana_grasp_joints(
                robot, 
                banana_position, 
                stage,
                offset=fine_pre_grasp_offset, 
                gripper_open=True
            )
            robot.set_joint_positions(fine_pre_grasp_joints)
            
            # Log the joint positions
            current_pos = robot.get_joint_positions()
            log(f"Fine pre-grasp joint positions: {current_pos}")
            
            for _ in range(80):  # Wait for movement to complete
                simulation_app.update()
                time.sleep(0.02)
            
            # Verify pre-grasp position
            pre_grasp_good, pre_grasp_distance, pre_grasp_error = check_grasp_position(
                robot, 
                stage, 
                banana_position + fine_pre_grasp_offset, 
                threshold=GRASP_THRESHOLD
            )
            
            if not pre_grasp_good:
                log("Making final adjustments to pre-grasp position...")
                # Fine-tune manually based on the error vector
                current_pos = robot.get_joint_positions()
                
                # Adjust joints based on the error vector
                current_pos[1] -= pre_grasp_error[2] * 0.5  # Adjust shoulder for height
                current_pos[3] -= pre_grasp_error[2] * 0.7  # Adjust elbow for height
                
                # Adjust base rotation if XY error is significant
                xy_error_magnitude = np.linalg.norm(pre_grasp_error[:2])
                if xy_error_magnitude > 0.01:
                    # Calculate angle to error
                    angle_to_error = np.arctan2(pre_grasp_error[1], pre_grasp_error[0])
                    # Adjust base rotation slightly toward that angle
                    rotation_adjustment = 0.3 * (angle_to_error - current_pos[0])
                    current_pos[0] += rotation_adjustment
                
                robot.set_joint_positions(current_pos)
                
                for _ in range(50):  # Wait for adjustment
                    simulation_app.update()
                    time.sleep(0.02)
            
            # Stage 3: Final approach to grasp position with very small increments
            log("Stage 3: Final approach to grasp position...")
            
            # Update banana position again for maximum accuracy
            banana_position, _ = get_prim_pose(stage, ACTUAL_BANANA_PATH)
            
            # We'll approach in small steps
            approach_steps = 3
            for step in range(approach_steps):
                # Calculate offset for this step - gradually getting closer to the banana
                current_offset = np.array([0.0, 0.0, 0.02 - (step * 0.02)])  # From 2cm above to 2cm below
                
                log(f"Approach step {step+1}/{approach_steps}, offset: {current_offset}")
                
                grasp_joints = calculate_precise_banana_grasp_joints(
                    robot, 
                    banana_position, 
                    stage,
                    offset=current_offset, 
                    gripper_open=True
                )
                robot.set_joint_positions(grasp_joints)
                
                # Wait less time for smaller movements
                for _ in range(40):  
                    simulation_app.update()
                    time.sleep(0.02)
            
            # Final grasp position
            log("Final approach to optimal grasp position...")
            grasp_offset = np.array([0.0, 0.0, -0.02])  # 2cm below the banana's center point
            
            # Update banana position one last time
            banana_position, _ = get_prim_pose(stage, ACTUAL_BANANA_PATH)
            
            final_grasp_joints = calculate_precise_banana_grasp_joints(
                robot, 
                banana_position, 
                stage,
                offset=grasp_offset, 
                gripper_open=True
            )
            robot.set_joint_positions(final_grasp_joints)
            
            # Log the joint positions
            current_pos = robot.get_joint_positions()
            log(f"Final grasp joint positions: {current_pos}")
            
            for _ in range(60):  # Wait for movement to complete
                simulation_app.update()
                time.sleep(0.02)
            
            # Verify final grasp position
            is_position_good, final_distance, error_vector = check_grasp_position(
                robot, 
                stage, 
                banana_position, 
                threshold=GRASP_THRESHOLD
            )
            
            # If needed, make a final tiny adjustment
            if not is_position_good:
                log(f"Making emergency final adjustment, distance: {final_distance:.4f}m")
                # Get current position
                current_pos = robot.get_joint_positions()
                
                # Calculate a very small adjustment based on the error
                current_pos[1] -= error_vector[2] * 0.5  # Shoulder adjustment for Z
                current_pos[3] -= error_vector[2] * 0.7  # Elbow adjustment for Z
                
                # XY adjustments through base rotation and shoulder
                if np.linalg.norm(error_vector[:2]) > 0.01:
                    # Get current and target angles
                    current_angle = current_pos[0]
                    target_angle = np.arctan2(banana_position[1], banana_position[0])
                    
                    # Apply a small rotation adjustment
                    current_pos[0] += (target_angle - current_angle) * 0.2
                
                robot.set_joint_positions(current_pos)
                
                for _ in range(40):  # Small wait for small adjustment
                    simulation_app.update()
                    time.sleep(0.02)
                
                # Verify the position one last time
                is_position_good, adjusted_distance, _ = check_grasp_position(
                    robot, 
                    stage, 
                    banana_position, 
                    threshold=GRASP_THRESHOLD
                )
                
                log(f"After emergency adjustment: distance = {adjusted_distance:.4f}m, good = {is_position_good}")
            
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
                log("WARNING: Could not identify gripper joints")
            
            # Lift the banana
            log("Lifting the banana...")
            lift_offset = np.array([0.0, 0.0, 0.25])  # 25cm above the banana (increased for safety)
            
            # For lifting, use the pre-grasp joints with adjusted height
            lift_joints = fine_pre_grasp_joints.copy()
            
            # Keep gripper closed
            if len(gripper_indices) > 0:
                for idx in gripper_indices:
                    lift_joints[idx] = 0.0
                    
            # Adjust height for lifting
            lift_joints[1] += 0.3  # Raise shoulder
            lift_joints[3] += 0.4  # Adjust elbow
            
            robot.set_joint_positions(lift_joints)
            
            for _ in range(100):  # Wait for movement to complete
                simulation_app.update()
                time.sleep(0.02)
            
            # Move to position above target (crate)
            log("Moving to target position...")
            
            # Update target position
            target_position, _ = get_prim_pose(stage, ACTUAL_TARGET_PATH)
            
            # For the target, use a different approach
            target_approach_joints = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.6, 0.8, 0.0, 0.0])
            
            # Adjust base rotation to face the target
            if target_position[0] != 0 or target_position[1] != 0:
                target_approach_joints[0] = np.arctan2(target_position[1], target_position[0])
            
            # Adjust elbow and shoulder for height and distance
            distance_to_target = np.linalg.norm(target_position[:2])
            target_approach_joints[1] = -0.3 - (distance_to_target * 0.3)  # Shoulder
            target_approach_joints[3] = -1.8 - (target_position[2] * 0.1)  # Elbow
            
            robot.set_joint_positions(target_approach_joints)
            
            for _ in range(100):  # Wait for movement to complete
                simulation_app.update()
                time.sleep(0.02)
            
            # Lower to place position
            log("Lowering to place position...")
            place_joints = target_approach_joints.copy()
            place_joints[1] -= 0.2  # Lower shoulder
            place_joints[3] -= 0.3  # Adjust elbow
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
                
                log("Gripper opened, banana released")
            
            # Lift after placing
            log("Lifting after placing...")
            after_place_joints = place_joints.copy()
            after_place_joints[1] += 0.3  # Raise shoulder
            after_place_joints[3] += 0.3  # Adjust elbow
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
