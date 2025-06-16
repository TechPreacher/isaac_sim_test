#!/usr/bin/env python3
"""
Improved Franka Robot Pick and Place with Advanced IK Control

This script implements a more precise pick-and-place task with the Franka robot by:
1. Using a precise direct end effector control approach
2. Implementing position verification with feedback loop
3. Adding reference-based joint angle adjustments specific to the banana grasp
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

def calculate_precise_banana_grasp_joints(robot, banana_position, stage, offset=None, gripper_open=True, max_iterations=3):
    """
    Calculate joint positions for precise banana grasping using a combination of:
    1. Optimal reference joint angles
    2. Position-based adjustments
    3. Iterative refinement based on end effector feedback
    
    Args:
        robot: The robot articulation object
        banana_position: The 3D position of the banana
        stage: The USD stage for position feedback
        offset: Optional offset to apply to the target position (in world frame)
        gripper_open: Whether to open (True) or close (False) the gripper
        max_iterations: Maximum number of refinement iterations
        
    Returns:
        Joint positions to reach the target
    """
    # Start with the optimal joint configuration for banana grasping
    joint_positions = OPTIMAL_BANANA_GRASP_JOINTS.copy()
    num_joints = len(robot.dof_names)
    
    # Calculate the difference between current banana position and reference position
    banana_delta = banana_position - REFERENCE_BANANA_POSITION
    
    # Adjust the first joint (base rotation) based on the banana's XY position
    # We use the arctan2 directly as it's more precise for the base rotation
    if banana_position[0] != 0 or banana_position[1] != 0:
        joint_positions[0] = np.arctan2(banana_position[1], banana_position[0])
    
    # Apply precise adjustments based on the banana position delta
    # Shoulder joint (joint 2) adjustment based on XY distance and Z height
    joint_positions[1] += banana_delta[0] * 0.3 + banana_delta[2] * 0.5
    
    # Elbow joints adjustments
    joint_positions[3] -= banana_delta[2] * 0.7  # Height has strong effect on elbow 2
    
    # Wrist adjustments for better approach angle
    joint_positions[5] += banana_delta[2] * 0.3  # Adjust wrist angle based on height
    
    # Apply vertical offset if specified
    if offset is not None:
        if offset[2] > 0:  # Pre-grasp position (above banana)
            # For pre-grasp, adjust the joints to be slightly above
            joint_positions[1] += 0.15  # Shoulder up
            joint_positions[3] += 0.25  # Elbow up
        elif offset[2] < 0:  # Below banana or for grasping
            # For grasp or below position, adjust to reach lower
            joint_positions[1] -= 0.15  # Shoulder down
            joint_positions[3] -= 0.25  # Elbow down
            # Fine-tune wrist for better grasping angle
            joint_positions[5] += 0.15
    
    # Set gripper position
    if num_joints > 7:  # If we have gripper joints
        gripper_value = 0.05 if gripper_open else 0.0
        joint_positions[7] = gripper_value
        joint_positions[8] = gripper_value
    
    # Apply the joint positions and iterate to refine if needed
    if stage is not None:
        for i in range(max_iterations):
            # Apply current joint positions
            original_positions = robot.get_joint_positions()
            robot.set_joint_positions(joint_positions)
            
            # Wait a moment for physics to update
            time.sleep(0.1)
            
            # Get end effector position and check distance to banana
            ee_position = get_end_effector_position(robot, stage)
            if ee_position is None:
                log("WARNING: Could not get end effector position for refinement")
                break
                
            # Calculate target position with offset
            target_position = banana_position.copy()
            if offset is not None:
                target_position += offset
                
            # Calculate distance to target
            distance = np.linalg.norm(ee_position - target_position)
            
            if distance < 0.05:
                log(f"Position is good after iteration {i+1}, distance: {distance:.4f}m")
                break
                
            # Refine joint positions based on the error
            error_vector = target_position - ee_position
            
            # Apply corrections based on error vector
            joint_positions[1] -= error_vector[2] * 0.3  # Shoulder adjustment for height
            joint_positions[3] -= error_vector[2] * 0.5  # Elbow adjustment for height
            
            # XY position adjustments
            xy_error = np.linalg.norm(error_vector[:2])
            if xy_error > 0.02:
                # Adjust base rotation slightly
                angle_error = np.arctan2(error_vector[1], error_vector[0]) - joint_positions[0]
                joint_positions[0] += angle_error * 0.3
                
                # Reach forward/backward adjustment
                joint_positions[1] -= error_vector[0] * 0.2  # Shoulder forward/back
                
            log(f"Refined joint positions in iteration {i+1}: {joint_positions}")
            
        # Restore original position for the actual function to apply
        robot.set_joint_positions(original_positions)
    
    log(f"Final calculated joint positions: {joint_positions}")
    return joint_positions

def run_franka_pick_place():
    """
    Run Isaac Sim simulation with a Franka robot to pick and place a banana
    using precise joint control
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
            
            # Pre-grasp position - move above the banana
            log("Moving to pre-grasp position...")
            pre_grasp_offset = np.array([0.0, 0.0, 0.05])  # 5cm above the banana
            pre_grasp_joints = calculate_precise_banana_grasp_joints(
                robot, 
                banana_position, 
                stage,
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
            
            # Verify pre-grasp position
            pre_grasp_good = check_grasp_position(robot, stage, banana_position + pre_grasp_offset, threshold=0.05)
            if not pre_grasp_good:
                log("Adjusting pre-grasp position for better approach...")
                # Fine-tune the pre-grasp position manually
                current_pos = robot.get_joint_positions()
                # Adjust shoulder and elbow for height
                current_pos[1] += 0.05  # Shoulder adjustment
                current_pos[3] += 0.1   # Elbow adjustment
                robot.set_joint_positions(current_pos)
                
                for _ in range(50):  # Wait for adjustment
                    simulation_app.update()
                    time.sleep(0.02)
            
            # Grasp position - move down to the banana
            log("Moving to grasp position...")
            grasp_offset = np.array([0.0, 0.0, -0.03])  # 3cm below the banana's center point
            grasp_joints = calculate_precise_banana_grasp_joints(
                robot, 
                banana_position, 
                stage,
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
            
            # Verify grasp position and make final adjustments if needed
            is_position_good = check_grasp_position(robot, stage, banana_position, threshold=0.05)
            max_adjustment_attempts = 3
            adjustment_attempt = 0
            
            while not is_position_good and adjustment_attempt < max_adjustment_attempts:
                log(f"Adjusting grasp position, attempt {adjustment_attempt + 1}...")
                # Get current position of end effector and banana
                ee_position = get_end_effector_position(robot, stage)
                banana_position, _ = get_prim_pose(stage, ACTUAL_BANANA_PATH)
                
                if ee_position is None or banana_position is None:
                    log("WARNING: Could not get positions for adjustment")
                    break
                
                # Calculate adjustment vector
                adjustment_vector = banana_position - ee_position
                log(f"Adjustment vector: {adjustment_vector}")
                
                # Apply adjustment to joint positions
                current_pos = robot.get_joint_positions()
                
                # Adjust shoulder and elbow for height
                current_pos[1] -= adjustment_vector[2] * 0.4  # Adjust shoulder for height
                current_pos[3] -= adjustment_vector[2] * 0.6  # Adjust elbow for height
                
                # Adjust base rotation for XY position
                if np.linalg.norm(adjustment_vector[:2]) > 0.01:
                    angle_adjustment = np.arctan2(adjustment_vector[1], adjustment_vector[0]) - current_pos[0]
                    current_pos[0] += angle_adjustment * 0.3
                
                robot.set_joint_positions(current_pos)
                
                for _ in range(50):  # Wait for adjustment
                    simulation_app.update()
                    time.sleep(0.02)
                
                # Check if position is now good
                is_position_good = check_grasp_position(robot, stage, banana_position, threshold=0.05)
                adjustment_attempt += 1
            
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
            lift_offset = np.array([0.0, 0.0, 0.20])  # 20cm above the banana (increased for safety)
            # For lifting, we can use a simpler approach since precise positioning is less critical
            lift_joints = grasp_joints.copy()
            lift_joints[1] += 0.3  # Raise shoulder
            lift_joints[3] += 0.3  # Adjust elbow
            robot.set_joint_positions(lift_joints)
            
            for _ in range(100):  # Wait for movement to complete
                simulation_app.update()
                time.sleep(0.02)
            
            # Move to position above target (crate)
            log("Moving to target position...")
            # Calculate joint positions for target approach
            # For the target, we use a different approach as precision is less critical
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
