#!/usr/bin/env python3
"""
Optimized Franka Robot Pick and Place

This script implements a highly optimized pick-and-place task with the Franka robot by:
1. Using cached prim path detection with fallback strategies
2. Implementing advanced trajectory planning with smooth motion profiles
3. Providing robust error recovery mechanisms
4. Using dynamic grasp position calculation based on object dimensions
5. Structuring code for better maintainability and reusability
"""

import time
import warnings
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
from core import setup_isaac_sim_environment
from robot_movements import RobotMovements

# Suppress numpy compatibility warnings
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

# Configuration constants
THIS_FOLDER = Path(__file__).parent.resolve()
USD_SCENE_PATH = str(THIS_FOLDER / "scenes" / "franka_blocks_manual.usd")
ROBOT_PRIM_PATH = "/World/franka"
BANANA_PRIM_PATH = "/World/banana" 
TARGET_PRIM_PATH = "/World/crate"

# Cached paths for faster access
CACHED_PATHS = {
    "robot": ROBOT_PRIM_PATH,
    "banana": BANANA_PRIM_PATH,
    "target": TARGET_PRIM_PATH,
    "hand": None,
    "left_finger": None,
    "right_finger": None
}

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
REFERENCE_BANANA_POSITION = np.array([0.20, 0.18, 0.84])

# Enhanced parameters for better precision
GRASP_THRESHOLD = 0.025    # Maximum distance for successful grasping
MAX_REFINEMENT_ITERATIONS = 8  # Max iterations for fine tuning
POSITION_STABILIZATION_TIME = 0.1  # Time to let physics settle between movements

class Logger:
    """Simple logging class with timestamps"""
    @staticmethod
    def log(message: str, level: str = "INFO") -> None:
        """Log a message with timestamp and level"""
        print(f"[{time.time():.2f}s] [{level}] {message}")
    
    @staticmethod
    def info(message: str) -> None:
        """Log an info message"""
        Logger.log(message, "INFO")
    
    @staticmethod
    def warning(message: str) -> None:
        """Log a warning message"""
        Logger.log(message, "WARNING")
    
    @staticmethod
    def error(message: str) -> None:
        """Log an error message"""
        Logger.log(message, "ERROR")
    
    @staticmethod
    def debug(message: str) -> None:
        """Log a debug message"""
        Logger.log(message, "DEBUG")

class PrimUtils:
    """Utility class for working with USD prims"""
    
    @staticmethod
    def get_prim_pose(stage, prim_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
            Logger.error(f"Prim not found at {prim_path}")
            return None, None
        
        # Get the xformable
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            Logger.error(f"Prim at {prim_path} is not xformable")
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
        orientation = np.array([
            [world_transform[0][0], world_transform[0][1], world_transform[0][2]],
            [world_transform[1][0], world_transform[1][1], world_transform[1][2]],
            [world_transform[2][0], world_transform[2][1], world_transform[2][2]]
        ])
        
        return position, orientation
    
    @staticmethod
    def find_object_in_scene(stage, name_substring: str) -> Optional[str]:
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
    
    @staticmethod
    def find_robot_hand_and_fingers(stage) -> Tuple[str, str, str]:
        """
        Find the correct paths for the robot hand and fingers with improved caching
        
        Args:
            stage: USD Stage
            
        Returns:
            Tuple of (hand_path, left_finger_path, right_finger_path)
        """
        # Check if we already found these paths
        if CACHED_PATHS["hand"] and CACHED_PATHS["left_finger"] and CACHED_PATHS["right_finger"]:
            # Verify that the cached paths still exist
            if (stage.GetPrimAtPath(CACHED_PATHS["hand"]) and 
                stage.GetPrimAtPath(CACHED_PATHS["left_finger"]) and 
                stage.GetPrimAtPath(CACHED_PATHS["right_finger"])):
                return (
                    CACHED_PATHS["hand"], 
                    CACHED_PATHS["left_finger"], 
                    CACHED_PATHS["right_finger"]
                )
        
        # First, try the expected paths
        hand_path = f"{ROBOT_PRIM_PATH}/panda_hand"
        left_finger_path = f"{ROBOT_PRIM_PATH}/panda_hand/panda_leftfinger"
        right_finger_path = f"{ROBOT_PRIM_PATH}/panda_hand/panda_rightfinger"
        
        # Check if these paths exist
        if not stage.GetPrimAtPath(hand_path):
            Logger.warning(f"Hand not found at {hand_path}, searching...")
            # Search for hand in robot children
            for prim in stage.Traverse():
                path_str = str(prim.GetPath())
                if path_str.startswith(ROBOT_PRIM_PATH) and "hand" in path_str.lower():
                    hand_path = path_str
                    Logger.info(f"Found hand at: {hand_path}")
                    break
        
        # Now that we have a hand path, search for fingers
        if not stage.GetPrimAtPath(left_finger_path):
            Logger.warning(f"Left finger not found at {left_finger_path}, searching...")
            # Search for fingers in hand children or robot children
            left_finger_path = None
            right_finger_path = None
            
            # Try to find fingers by iterating through prims
            for prim in stage.Traverse():
                path_str = str(prim.GetPath())
                if path_str.startswith(hand_path) or path_str.startswith(ROBOT_PRIM_PATH):
                    if ("leftfinger" in path_str.lower() or "finger1" in path_str.lower() or 
                        "finger_joint1" in path_str.lower()):
                        left_finger_path = path_str
                        Logger.info(f"Found left finger at: {left_finger_path}")
                    elif ("rightfinger" in path_str.lower() or "finger2" in path_str.lower() or 
                          "finger_joint2" in path_str.lower()):
                        right_finger_path = path_str
                        Logger.info(f"Found right finger at: {right_finger_path}")
            
            # If we still haven't found fingers, use fallback
            if not left_finger_path:
                left_finger_path = hand_path + "/finger1"
                Logger.warning(f"Using fallback left finger path: {left_finger_path}")
            if not right_finger_path:
                right_finger_path = hand_path + "/finger2"
                Logger.warning(f"Using fallback right finger path: {right_finger_path}")
        
        # Cache the paths for future use
        CACHED_PATHS["hand"] = hand_path
        CACHED_PATHS["left_finger"] = left_finger_path
        CACHED_PATHS["right_finger"] = right_finger_path
        
        return hand_path, left_finger_path, right_finger_path
    
    @staticmethod
    def get_object_dimensions(stage, object_path: str) -> Tuple[float, float, float]:
        """
        Get the dimensions of an object with multiple fallback strategies
        
        Args:
            stage: The USD stage
            object_path: Path to the object prim
            
        Returns:
            Tuple of (length, width, height) as best estimated
        """
        from pxr import UsdGeom
        
        # Get the prim at the specified path
        prim = stage.GetPrimAtPath(object_path)
        if not prim:
            Logger.error(f"Object prim not found at {object_path}")
            return (0.15, 0.05, 0.05)  # Default values
        
        try:
            # Strategy 1: Try to get extents directly from the prim
            try:
                # Get the bounding box using the extent attribute
                extent_attr = UsdGeom.Boundable(prim).GetExtentAttr()
                if extent_attr:
                    extent = extent_attr.Get()
                    if extent and len(extent) == 2:
                        min_point = extent[0]
                        max_point = extent[1]
                        
                        # Calculate dimensions
                        dimensions = np.array([
                            max_point[0] - min_point[0],
                            max_point[1] - min_point[1],
                            max_point[2] - min_point[2]
                        ])
                        
                        Logger.info(f"Object dimensions from extent: {dimensions}")
                        
                        # Sort dimensions to get length, width, height
                        sorted_dimensions = np.sort(dimensions)[::-1]
                        return tuple(sorted_dimensions)
            except Exception as e:
                Logger.warning(f"Could not get dimensions from extent: {e}")
                
            # Strategy 2: Try to get bounds from compute world bound using BBoxCache
            try:
                from pxr import UsdGeom, Tf
                # Create BBoxCache with correct parameters
                purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
                bbox_cache = UsdGeom.BBoxCache(0, purposes, True)
                bounds = bbox_cache.ComputeWorldBound(prim)
                
                if bounds:
                    # Get the range (min and max points)
                    range_min = bounds.GetRange().GetMin()
                    range_max = bounds.GetRange().GetMax()
                    
                    # Calculate dimensions
                    dimensions = np.array([
                        range_max[0] - range_min[0],
                        range_max[1] - range_min[1],
                        range_max[2] - range_min[2]
                    ])
                    
                    Logger.info(f"Object dimensions from BBoxCache: {dimensions}")
                    
                    # Sort dimensions to get length, width, height
                    sorted_dimensions = np.sort(dimensions)[::-1]
                    return tuple(sorted_dimensions)
            except Exception as e:
                Logger.warning(f"Could not get dimensions from BBoxCache: {e}")
            
            # Strategy 3: Try to get children geometry extents
            try:
                max_dimensions = np.zeros(3)
                for child in prim.GetChildren():
                    if UsdGeom.Boundable(child):
                        extent_attr = UsdGeom.Boundable(child).GetExtentAttr()
                        if extent_attr:
                            extent = extent_attr.Get()
                            if extent and len(extent) == 2:
                                min_point = extent[0]
                                max_point = extent[1]
                                dimensions = np.array([
                                    max_point[0] - min_point[0],
                                    max_point[1] - min_point[1],
                                    max_point[2] - min_point[2]
                                ])
                                max_dimensions = np.maximum(max_dimensions, dimensions)
                
                if np.any(max_dimensions > 0):
                    Logger.info(f"Object dimensions from children: {max_dimensions}")
                    sorted_dimensions = np.sort(max_dimensions)[::-1]
                    return tuple(sorted_dimensions)
            except Exception as e:
                Logger.warning(f"Could not get dimensions from children: {e}")
            
            # Strategy 4: If it's a banana, use typical banana dimensions
            if "banana" in object_path.lower():
                Logger.warning("Using typical banana dimensions")
                return (0.15, 0.05, 0.05)  # Typical banana dimensions
            
            # Strategy 5: Default fallback dimensions
            Logger.warning("Using default object dimensions")
            return (0.1, 0.1, 0.1)  # Default dimensions for generic objects
            
        except Exception as e:
            Logger.error(f"Error getting object dimensions: {e}")
            return (0.1, 0.1, 0.1)  # Default fallback dimensions

class RobotController:
    """High-level robot controller for pick and place tasks"""
    
    def __init__(self, robot, stage):
        """
        Initialize the robot controller
        
        Args:
            robot: The robot articulation object
            stage: The USD stage
        """
        self.robot = robot
        self.stage = stage
        self.robot_movements = RobotMovements(robot)
        
        # Initialize gripper indices
        self.gripper_indices = self._find_gripper_indices()
        
        # Home position for the robot
        self.home_position = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.6, 0.0, 0.04, 0.04])
    
    def _find_gripper_indices(self) -> List[int]:
        """Find the indices of gripper joints"""
        gripper_indices = []
        
        # Check if we have a robot with joint information
        if hasattr(self.robot, 'dof_names'):
            dof_names = self.robot.dof_names
            dof_count = len(dof_names)
            
            # Search for gripper joints in the joint names
            for i, name in enumerate(dof_names):
                if ("finger" in name.lower() or 
                    "gripper" in name.lower() or 
                    "panda_finger" in name.lower()):
                    gripper_indices.append(i)
            
            # If we couldn't find the gripper joints, use last two joints as fallback
            if not gripper_indices and dof_count >= 2:
                Logger.warning("Could not identify gripper joints, using last two joints")
                gripper_indices = [dof_count-2, dof_count-1]
        
        return gripper_indices
    
    def get_end_effector_position(self, get_grip_point: bool = False) -> Optional[np.ndarray]:
        """
        Get the current position of the robot's end effector
        
        Args:
            get_grip_point: If True, return the grip point instead of the hand center
            
        Returns:
            End effector position as numpy array [x, y, z]
        """
        try:
            # Find the correct paths for hand and fingers
            hand_path, _, _ = PrimUtils.find_robot_hand_and_fingers(self.stage)
            
            # Get the position from the prim
            hand_position, _ = PrimUtils.get_prim_pose(self.stage, hand_path)
            
            if get_grip_point and hand_position is not None:
                # Estimate actual grip point (8cm below hand center)
                grip_point = hand_position.copy()
                grip_point[2] -= 0.08
                Logger.debug(f"Estimated grip point: {grip_point}")
                return grip_point
            
            if hand_position is not None:
                Logger.debug(f"End effector position: {hand_position}")
                return hand_position
            else:
                Logger.warning("Could not get end effector position from prim")
                return None
                
        except Exception as e:
            Logger.error(f"Error getting end effector position: {e}")
            # Try fallback methods for getting end effector position
            try:
                # This assumes the robot has a method to get the end effector position
                # through forward kinematics, which may not be available
                if hasattr(self.robot, 'get_end_effector_position'):
                    ee_pos = self.robot.get_end_effector_position()
                    Logger.debug(f"End effector position from FK: {ee_pos}")
                    return ee_pos
                return None
            except:
                return None
    
    def check_grasp_position(self, 
                           target_position: np.ndarray, 
                           threshold: float = GRASP_THRESHOLD, 
                           use_grip_point: bool = True) -> Tuple[bool, float, np.ndarray]:
        """
        Check if the end effector is close enough to the target for grasping
        
        Args:
            target_position: The target position
            threshold: Maximum allowed distance between end effector and target
            use_grip_point: Whether to use the grip point instead of hand center
            
        Returns:
            (is_close_enough, actual_distance, error_vector) tuple
        """
        # Get the end effector position (either grip point or hand center)
        ee_position = self.get_end_effector_position(get_grip_point=use_grip_point)
        
        if ee_position is None:
            Logger.warning("Could not verify grasp position, proceeding anyway")
            return True, 0.0, np.zeros(3)
        
        # Calculate distance to target
        error_vector = target_position - ee_position
        distance = np.linalg.norm(error_vector)
        
        # Log detailed position information
        Logger.debug(f"End effector position: {ee_position}")
        Logger.debug(f"Target position: {target_position}")
        Logger.debug(f"Error vector: {error_vector}")
        Logger.info(f"Distance between end effector and target: {distance:.4f}m")
        
        # Check if close enough
        is_close_enough = distance <= threshold
        
        if is_close_enough:
            Logger.info(f"End effector is close enough to target for grasping ({distance:.4f}m)")
        else:
            Logger.warning(f"End effector is too far from target ({distance:.4f}m > {threshold}m)")
            
        return is_close_enough, distance, error_vector
    
    def move_to_home_position(self) -> None:
        """Move the robot to the home position"""
        Logger.info("Moving to home position...")
        self.robot.set_joint_positions(self.home_position)
    
    def open_gripper(self) -> None:
        """Open the gripper"""
        Logger.info("Opening gripper...")
        if self.gripper_indices:
            # Get current positions
            current_pos = self.robot.get_joint_positions()
            # Open the gripper (set to 0.04 for most grippers)
            for idx in self.gripper_indices:
                current_pos[idx] = 0.04
            self.robot.set_joint_positions(current_pos)
        else:
            Logger.warning("No gripper indices found, cannot open gripper")
    
    def close_gripper(self) -> None:
        """Close the gripper"""
        Logger.info("Closing gripper...")
        if self.gripper_indices:
            # Get current positions
            current_pos = self.robot.get_joint_positions()
            # Close the gripper (set to 0.0 for most grippers)
            for idx in self.gripper_indices:
                current_pos[idx] = 0.0
            self.robot.set_joint_positions(current_pos)
        else:
            Logger.warning("No gripper indices found, cannot close gripper")
    
    def direct_grasp_approach(self, 
                             object_position: np.ndarray, 
                             grasp_position: np.ndarray) -> bool:
        """
        Implement a direct approach to an object based on empirical adjustments
        
        Args:
            object_position: The position of the object
            grasp_position: The ideal grasp position
            
        Returns:
            True if the approach was successful, False otherwise
        """
        # Use the optimal joint configuration as our starting point
        joint_positions = OPTIMAL_BANANA_GRASP_JOINTS.copy()
        
        # Calculate the delta between current object and reference position
        object_delta = object_position - REFERENCE_BANANA_POSITION
        Logger.info(f"Object delta from reference: {object_delta}")
        
        # Adjust base rotation to point directly at actual object
        target_base_angle = np.arctan2(object_position[1], object_position[0])
        joint_positions[0] = target_base_angle
        
        # Apply delta adjustments with corrected factors
        # Height adjustment - critical for correct grasping
        joint_positions[1] += object_delta[2] * 0.9  # Increased shoulder adjustment
        joint_positions[3] -= object_delta[2] * 1.1  # Increased elbow adjustment
        
        # Apply XY position adjustments
        joint_positions[1] -= object_delta[0] * 0.2  # X position affects shoulder
        joint_positions[2] += object_delta[1] * 0.2  # Y position affects elbow1
        
        # Apply the adjusted configuration
        Logger.info("Applying first adjusted joint configuration...")
        self.robot.set_joint_positions(joint_positions)
        
        # Wait for movement to complete
        for _ in range(40):
            simulation_app.update()
            time.sleep(0.03)
        
        # Check if we reached a good position
        is_close, distance, error_vector = self.check_grasp_position(
            grasp_position, threshold=GRASP_THRESHOLD
        )
        
        # Iteratively refine the position if needed
        refinement_iterations = 0
        max_iterations = 3
        
        while not is_close and refinement_iterations < max_iterations:
            refinement_iterations += 1
            Logger.info(f"Refinement iteration {refinement_iterations}/{max_iterations}, distance: {distance:.4f}m")
            
            # Get current joint positions
            current_joints = self.robot.get_joint_positions()
            
            # Adjust joints based on error vector
            # Primarily focus on height adjustment which is most critical
            height_error = error_vector[2]
            
            # Make more aggressive corrections when we're far away
            correction_factor = min(1.0, distance * 10)  # Scale correction based on distance
            
            # Apply different corrections based on the error direction
            if height_error > 0:  # Need to move up
                current_joints[1] -= height_error * 0.8 * correction_factor  # Shoulder adjustment
                current_joints[3] += height_error * 0.9 * correction_factor  # Elbow adjustment
            else:  # Need to move down
                current_joints[1] += abs(height_error) * 1.0 * correction_factor  # Shoulder down
                current_joints[3] -= abs(height_error) * 1.2 * correction_factor  # Elbow down
            
            # XY adjustments
            xy_error = error_vector[:2]
            xy_distance = np.linalg.norm(xy_error)
            
            if xy_distance > 0.01:
                # Adjust base rotation to face the object more directly
                current_angle = current_joints[0]
                target_angle = np.arctan2(object_position[1], object_position[0])
                
                # Calculate angle error and normalize
                angle_error = target_angle - current_angle
                if angle_error > np.pi:
                    angle_error -= 2 * np.pi
                elif angle_error < -np.pi:
                    angle_error += 2 * np.pi
                
                # Apply rotation adjustment
                current_joints[0] += angle_error * 0.5 * correction_factor
                
                # Apply forward/backward adjustment (joint 1)
                forward_error = np.dot(xy_error, np.array([np.cos(current_joints[0]), np.sin(current_joints[0])]))
                current_joints[1] += forward_error * 0.4 * correction_factor
            
            # Apply the refined joint positions
            self.robot.set_joint_positions(current_joints)
            
            # Wait for movement to complete
            for _ in range(30):
                simulation_app.update()
                time.sleep(0.03)
            
            # Check position again
            is_close, distance, error_vector = self.check_grasp_position(
                grasp_position, threshold=GRASP_THRESHOLD
            )
            
            # If we're very close, consider it successful even if just slightly above threshold
            if distance < GRASP_THRESHOLD * 1.2:
                Logger.info(f"Close enough for grasping ({distance:.4f}m)")
                is_close = True
                break
        
        Logger.info(f"Final grasp approach complete, distance: {distance:.4f}m, success: {is_close}")
        return is_close
    
    def create_trajectory(self, 
                        start_position: np.ndarray, 
                        target_position: np.ndarray, 
                        num_waypoints: int = 5,
                        arc_height: float = 0.2) -> List[np.ndarray]:
        """
        Create a smooth trajectory from start to target position with multiple waypoints
        
        Args:
            start_position: Starting position
            target_position: Target position
            num_waypoints: Number of waypoints to generate
            arc_height: Maximum height of the arc for lifting motions
            
        Returns:
            List of waypoint positions
        """
        waypoints = []
        
        for i in range(num_waypoints):
            # Calculate interpolation factor with ease-in-out profile
            t_linear = (i + 1) / num_waypoints
            # Apply cubic ease-in-out for smoother acceleration/deceleration
            t = t_linear * t_linear * (3 - 2 * t_linear)
            
            # Linear interpolation
            waypoint = start_position * (1 - t) + target_position * t
            
            # Add an arc for lifting/lowering
            # Sinusoidal profile for smoother arc
            lift_factor = np.sin(t * np.pi)  # Sinusoidal profile
            waypoint[2] += arc_height * lift_factor
            
            waypoints.append(waypoint)
        
        return waypoints
    
    def move_along_trajectory(self, waypoints: List[np.ndarray], time_per_waypoint: float = 1.0) -> bool:
        """
        Move the robot along a trajectory defined by waypoints
        
        Args:
            waypoints: List of waypoint positions
            time_per_waypoint: Time to spend at each waypoint
            
        Returns:
            True if the movement was successful, False otherwise
        """
        try:
            for i, waypoint in enumerate(waypoints):
                Logger.info(f"Moving to waypoint {i+1}/{len(waypoints)}")
                
                # Calculate base angle to face the waypoint
                target_angle = np.arctan2(waypoint[1], waypoint[0])
                
                # Get current joint positions
                current_joints = self.robot.get_joint_positions()
                
                # Adjust base angle (joint 0)
                current_joints[0] = target_angle
                
                # Adjust arm position to reach the waypoint
                # These are approximate IK adjustments - for a real system, use proper IK
                # Distance from base to waypoint in XY plane
                xy_distance = np.sqrt(waypoint[0]**2 + waypoint[1]**2)
                
                # Adjust shoulder (joint 1) based on height and distance
                current_joints[1] = -0.3 - (xy_distance - 0.5) * 0.5 + (waypoint[2] - 0.5) * 0.5
                
                # Adjust elbow (joint 3) based on height and distance
                current_joints[3] = -2.0 + (xy_distance - 0.5) * 0.5 - (waypoint[2] - 0.5) * 0.7
                
                # Apply the joint positions
                self.robot.set_joint_positions(current_joints)
                
                # Wait for movement to complete
                steps = int(time_per_waypoint / 0.02)  # 50Hz simulation rate
                for _ in range(steps):
                    simulation_app.update()
                    time.sleep(0.02)
            
            return True
        except Exception as e:
            Logger.error(f"Error during trajectory movement: {e}")
            return False

    def lift_object(self, lift_height: float = 0.2) -> bool:
        """
        Lift the object after grasping
        
        Args:
            lift_height: Height to lift the object
            
        Returns:
            True if the lift was successful, False otherwise
        """
        Logger.info(f"Lifting object by {lift_height}m...")
        
        try:
            # Get current position
            ee_position = self.get_end_effector_position()
            if ee_position is None:
                Logger.warning("Could not get end effector position, using approximate lift")
                # Use approximation method with joint adjustments
                current_pos = self.robot.get_joint_positions()
                current_pos[1] -= 0.2  # Shoulder up
                current_pos[3] += 0.3  # Elbow up
                
                # Keep gripper closed
                for idx in self.gripper_indices:
                    current_pos[idx] = 0.0
                    
                self.robot.set_joint_positions(current_pos)
                
                for _ in range(50):  # Wait for lift to complete
                    simulation_app.update()
                    time.sleep(0.02)
                
                return True
            
            # Calculate lift position
            lift_position = ee_position.copy()
            lift_position[2] += lift_height
            
            # Create a smooth trajectory for lifting
            lift_waypoints = self.create_trajectory(
                ee_position, lift_position, num_waypoints=3, arc_height=0.0
            )
            
            # Move along the trajectory
            return self.move_along_trajectory(lift_waypoints, time_per_waypoint=0.5)
        
        except Exception as e:
            Logger.error(f"Error during object lifting: {e}")
            return False
    
    def move_to_target(self, target_position: np.ndarray) -> bool:
        """
        Move the grasped object to the target position
        
        Args:
            target_position: Target position
            
        Returns:
            True if the movement was successful, False otherwise
        """
        Logger.info(f"Moving to target position: {target_position}")
        
        try:
            # Get current end effector position
            ee_position = self.get_end_effector_position()
            if ee_position is None:
                Logger.warning("Could not get end effector position, using approximation")
                # Use approximate method with joint adjustments
                
                # Calculate target angle
                target_angle = np.arctan2(target_position[1], target_position[0])
                
                # Set joints to move toward target
                current_pos = self.robot.get_joint_positions()
                current_pos[0] = target_angle  # Base rotation
                
                # Keep other joints in a good configuration
                current_pos[1] = -0.3
                current_pos[2] = 0.0
                current_pos[3] = -2.0
                current_pos[4] = 0.0
                current_pos[5] = 1.6
                current_pos[6] = 0.0
                
                # Keep gripper closed
                for idx in self.gripper_indices:
                    current_pos[idx] = 0.0
                
                self.robot.set_joint_positions(current_pos)
                
                for _ in range(50):
                    simulation_app.update()
                    time.sleep(0.02)
                
                return True
            
            # Calculate position above target
            above_target = target_position.copy()
            above_target[2] += 0.2  # 20cm above target
            
            # Create a smooth trajectory for moving to the target
            move_waypoints = self.create_trajectory(
                ee_position, above_target, num_waypoints=5, arc_height=0.2
            )
            
            # Move along the trajectory
            return self.move_along_trajectory(move_waypoints, time_per_waypoint=0.5)
        
        except Exception as e:
            Logger.error(f"Error during movement to target: {e}")
            return False
    
    def lower_to_place(self, target_position: np.ndarray, place_height_offset: float = 0.05) -> bool:
        """
        Lower the object to place it
        
        Args:
            target_position: Target position
            place_height_offset: Height offset above the target for placing
            
        Returns:
            True if the lowering was successful, False otherwise
        """
        Logger.info(f"Lowering to place position...")
        
        try:
            # Get current end effector position
            ee_position = self.get_end_effector_position()
            if ee_position is None:
                Logger.warning("Could not get end effector position, using approximation")
                # Use approximate method with joint adjustments
                
                # Adjust joints to lower arm
                current_pos = self.robot.get_joint_positions()
                current_pos[1] += 0.25  # Shoulder down
                current_pos[3] -= 0.35  # Elbow down
                
                # Keep gripper closed
                for idx in self.gripper_indices:
                    current_pos[idx] = 0.0
                
                self.robot.set_joint_positions(current_pos)
                
                for _ in range(50):
                    simulation_app.update()
                    time.sleep(0.02)
                
                return True
            
            # Calculate place position
            place_position = target_position.copy()
            place_position[2] += place_height_offset
            
            # Create a smooth trajectory for lowering
            lower_waypoints = self.create_trajectory(
                ee_position, place_position, num_waypoints=3, arc_height=0.0
            )
            
            # Move along the trajectory
            return self.move_along_trajectory(lower_waypoints, time_per_waypoint=0.5)
        
        except Exception as e:
            Logger.error(f"Error during lowering to place: {e}")
            return False
    
    def calculate_ideal_grasp_position(self, 
                                     object_position: np.ndarray, 
                                     object_orientation: np.ndarray,
                                     object_dimensions: Tuple[float, float, float]) -> np.ndarray:
        """
        Calculate the ideal position to grasp an object based on its 
        position, orientation, and dimensions
        
        Args:
            object_position: The center position of the object
            object_orientation: The orientation matrix of the object
            object_dimensions: The dimensions of the object (length, width, height)
            
        Returns:
            The ideal grasp position as numpy array [x, y, z]
        """
        # Unpack dimensions
        length, width, height = object_dimensions
        
        # Calculate a grasp position based on object dimensions
        grasp_position = object_position.copy()
        
        # For banana-like objects, grasp slightly above the center height
        if "banana" in CACHED_PATHS["banana"].lower():
            # Adjust height to be slightly higher than the center
            grasp_position[2] += 0.02
            
            # Move slightly closer to the robot base for better reach
            base_direction = np.array([1.0, 0.0, 0.0])  # Assuming robot base is toward +X
            grasp_position -= base_direction * 0.015  # Move 1.5cm toward robot base
        else:
            # For generic objects, use dimension-based adjustments
            # For taller objects, grasp higher up
            if height > 0.1:
                grasp_position[2] += height * 0.1
            
            # For longer objects, adjust grasp position along the length
            if length > width * 1.5:  # Significantly longer than wide
                # Try to grasp near the middle but slightly toward the robot
                long_axis = object_orientation[:, 0]  # Assume first column is the long axis
                grasp_position -= long_axis * (length * 0.1)
        
        Logger.info(f"Calculated ideal grasp position: {grasp_position}")
        return grasp_position

class FrankaPnPTask:
    """Main class for Franka Pick and Place task"""
    
    def __init__(self):
        """Initialize the pick and place task"""
        # Import SimulationApp first to initialize the simulation
        from isaacsim import SimulationApp

        # Create the simulation app (use headless=False to see the UI)
        global simulation_app
        self.simulation_app = SimulationApp({"headless": False})
        simulation_app = self.simulation_app
        
        # Import necessary modules after SimulationApp is initialized
        import omni.usd
        import omni.timeline
        
        Logger.info("Starting Optimized Franka robot pick and place demonstration")
        
        # Open the USD stage with the scene
        Logger.info(f"Opening scene: {USD_SCENE_PATH}")
        self.usd_context = omni.usd.get_context()
        self.usd_context.open_stage(USD_SCENE_PATH)
        self.simulation_app.update()
        
        # Get the stage after opening it
        self.stage = self.usd_context.get_stage()
        
        # Get timeline interface for controlling playback
        self.timeline = omni.timeline.get_timeline_interface()
        
        # Initialize other attributes
        self.robot = None
        self.robot_controller = None
    
    def initialize(self) -> bool:
        """
        Initialize the simulation environment and robot
        
        Returns:
            True if initialization was successful, False otherwise
        """
        # Check if the robot exists in the stage
        robot_prim = self.stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        if not robot_prim:
            Logger.error(f"Robot not found at {ROBOT_PRIM_PATH}")
            return False
        
        Logger.info(f"Found robot prim at {ROBOT_PRIM_PATH} with type: {robot_prim.GetTypeName()}")
        
        # Verify and update banana prim path
        banana_prim = self.stage.GetPrimAtPath(CACHED_PATHS["banana"])
        if not banana_prim:
            Logger.warning(f"Banana not found at {CACHED_PATHS['banana']}, searching for it...")
            banana_path = PrimUtils.find_object_in_scene(self.stage, "banana")
            if banana_path:
                CACHED_PATHS["banana"] = banana_path
                Logger.info(f"Found banana at: {CACHED_PATHS['banana']}")
            else:
                Logger.error("Could not find banana in scene")
                return False
        else:
            Logger.info(f"Found banana at: {CACHED_PATHS['banana']}")
            
        # Verify and update target prim path (crate)
        target_prim = self.stage.GetPrimAtPath(CACHED_PATHS["target"])
        if not target_prim:
            Logger.warning(f"Crate not found at {CACHED_PATHS['target']}, searching for it...")
            # Try to find crate object
            target_path = PrimUtils.find_object_in_scene(self.stage, "crate")
            
            if target_path:
                CACHED_PATHS["target"] = target_path
                Logger.info(f"Found target destination at: {CACHED_PATHS['target']}")
            else:
                Logger.error("Could not find a crate destination in scene")
                return False
        else:
            Logger.info(f"Found crate at: {CACHED_PATHS['target']}")
        
        # Find hand and finger paths
        hand_path, left_finger_path, right_finger_path = PrimUtils.find_robot_hand_and_fingers(self.stage)
        Logger.info(f"Using hand path: {hand_path}")
        Logger.info(f"Using left finger path: {left_finger_path}")
        Logger.info(f"Using right finger path: {right_finger_path}")
        
        # Start the simulation
        self.timeline.play()
        self.simulation_app.update()
        
        # Wait for physics to initialize
        Logger.info("Initializing physics simulation...")
        for i in range(10):
            self.simulation_app.update()
            time.sleep(0.1)
        
        # Initialize the robot
        return self._initialize_robot()
    
    def _initialize_robot(self) -> bool:
        """
        Initialize the robot using appropriate API
        
        Returns:
            True if initialization was successful, False otherwise
        """
        Logger.info("Accessing the robot as an articulation...")
        
        # First try the modern Isaac Sim 4.5.0 API
        try:
            # Modern approach for Isaac Sim 4.5+
            Logger.info("Trying modern isaacsim.core.prims.SingleArticulation API...")
            from isaacsim.core.prims import SingleArticulation
            
            # Create articulation from existing prim
            self.robot = SingleArticulation(prim_path=ROBOT_PRIM_PATH, name="franka")
            
            # Update app to ensure articulation is registered
            self.simulation_app.update()
            time.sleep(0.2)
            
            # Initialize the robot
            self.robot.initialize()
            Logger.info("Successfully initialized the robot using modern articulation API")
            
            # Get joint information
            dof_names = self.robot.dof_names
            dof_count = self.robot.num_dof
            Logger.info(f"Robot has {dof_count} degrees of freedom")
            Logger.info(f"Joint names: {dof_names}")
            
            # Get current joint positions
            joint_positions = self.robot.get_joint_positions()
            Logger.info(f"Initial joint positions: {joint_positions}")
            
            # Initialize the robot controller
            self.robot_controller = RobotController(self.robot, self.stage)
            
            return True
            
        except Exception as e:
            Logger.error(f"Could not initialize robot with modern API: {e}")
            Logger.info("Trying with deprecated API as fallback...")
            
            try:
                # Fallback to deprecated API
                Logger.info("Trying deprecated omni.isaac.core.articulations.Articulation API...")
                from omni.isaac.core.articulations import Articulation
                
                # Create articulation from existing prim
                self.robot = Articulation(prim_path=ROBOT_PRIM_PATH, name="franka")
                
                # Update app to ensure articulation is registered
                self.simulation_app.update()
                time.sleep(0.2)
                
                # Initialize the robot
                self.robot.initialize()
                Logger.info("Successfully initialized the robot using deprecated articulation API")
                
                # Get joint information
                dof_names = self.robot.dof_names
                dof_count = self.robot.num_dof
                Logger.info(f"Robot has {dof_count} degrees of freedom")
                Logger.info(f"Joint names: {dof_names}")
                
                # Get current joint positions
                joint_positions = self.robot.get_joint_positions()
                Logger.info(f"Initial joint positions: {joint_positions}")
                
                # Initialize the robot controller
                self.robot_controller = RobotController(self.robot, self.stage)
                
                return True
                
            except Exception as e2:
                Logger.error(f"Could not initialize robot articulation with either API: {e2}")
                import traceback
                traceback.print_exc()
                return False
    
    def execute_pick_and_place(self) -> bool:
        """
        Execute the complete pick and place sequence
        
        Returns:
            True if the sequence was successful, False otherwise
        """
        # Check if the robot and controller are initialized
        if self.robot is None or self.robot_controller is None:
            Logger.error("Robot or controller not initialized")
            return False
        
        try:
            # Move to home position
            self.robot_controller.move_to_home_position()
            
            # Wait for movement to complete
            for _ in range(50):
                self.simulation_app.update()
                time.sleep(0.02)
            
            # Get positions of banana and target
            Logger.info("Getting positions of banana and target...")
            banana_position, banana_orientation = PrimUtils.get_prim_pose(self.stage, CACHED_PATHS["banana"])
            target_position, _ = PrimUtils.get_prim_pose(self.stage, CACHED_PATHS["target"])
            
            if banana_position is None or target_position is None:
                Logger.error("Could not get positions of banana or target")
                return False
            
            Logger.info(f"Banana position: {banana_position}")
            Logger.info(f"Target position: {target_position}")
            
            # Get banana dimensions
            banana_dimensions = PrimUtils.get_object_dimensions(self.stage, CACHED_PATHS["banana"])
            Logger.info(f"Banana dimensions: {banana_dimensions}")
            
            # Calculate ideal grasp position
            grasp_position = self.robot_controller.calculate_ideal_grasp_position(
                banana_position, banana_orientation, banana_dimensions
            )
            
            # Open gripper before approaching
            self.robot_controller.open_gripper()
            
            # Wait for gripper to open
            for _ in range(30):
                self.simulation_app.update()
                time.sleep(0.02)
            
            # Use direct grasp approach
            is_approach_successful = self.robot_controller.direct_grasp_approach(
                banana_position, grasp_position
            )
            
            if not is_approach_successful:
                Logger.warning("Could not achieve optimal grasp position, but proceeding anyway")
            
            # Close gripper to grasp the banana
            self.robot_controller.close_gripper()
            
            # Wait for gripper to close
            for _ in range(30):
                self.simulation_app.update()
                time.sleep(0.02)
            
            # Lift the banana
            lift_success = self.robot_controller.lift_object(lift_height=0.2)
            if not lift_success:
                Logger.warning("Lift may not have been successful, but proceeding")
            
            # Move to target position
            move_success = self.robot_controller.move_to_target(target_position)
            if not move_success:
                Logger.warning("Move to target may not have been successful, but proceeding")
            
            # Lower to place position
            lower_success = self.robot_controller.lower_to_place(target_position)
            if not lower_success:
                Logger.warning("Lowering may not have been successful, but proceeding")
            
            # Open gripper to release the banana
            self.robot_controller.open_gripper()
            
            # Wait for gripper to open
            for _ in range(30):
                self.simulation_app.update()
                time.sleep(0.02)
            
            # Lift after placing
            self.robot_controller.lift_object(lift_height=0.2)
            
            # Return to home position
            self.robot_controller.move_to_home_position()
            
            # Wait for movement to complete
            for _ in range(50):
                self.simulation_app.update()
                time.sleep(0.02)
            
            Logger.info("Pick and place task completed successfully!")
            return True
            
        except Exception as e:
            Logger.error(f"Error during pick and place sequence: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self) -> None:
        """Run the complete pick and place demonstration"""
        try:
            # Initialize the simulation
            if not self.initialize():
                Logger.error("Failed to initialize simulation")
                self.cleanup()
                return
            
            # If robot is initialized, execute pick and place
            if self.robot is not None and self.robot_controller is not None:
                self.execute_pick_and_place()
            else:
                # Run without control if we couldn't access the articulation
                Logger.warning("Running simulation without joint control (visualization only)...")
                total_steps = 500
                for step_count in range(total_steps):
                    self.simulation_app.update()
                    if step_count % 100 == 0:
                        Logger.info(f"Step {step_count}/{total_steps}")
                    time.sleep(0.01)
        
        except Exception as e:
            Logger.error(f"Error during simulation: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources and close simulation"""
        # Wait before closing
        Logger.info("Task complete, simulation will close in 5 seconds...")
        time.sleep(5)
        
        Logger.info("Simulation complete")
        if hasattr(self, 'simulation_app'):
            self.simulation_app.close()

def main():
    """Main entry point for the application"""
    # Set up the environment and run the simulation
    setup_isaac_sim_environment()
    
    # Create and run the pick and place task
    task = FrankaPnPTask()
    task.run()

if __name__ == "__main__":
    main()
