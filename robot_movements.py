# filepath: c:\Users\sascha\Code\90731-KUKA-DataCenterRoboticsChallenge\spikes\isaac-sim-environment\robot_movements.py
import warnings
# Suppress warnings about NumPy version mismatches
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

# Import NumPy with error handling
try:
    import numpy as np
except ImportError as e:
    print(f"Error importing NumPy: {e}")
    print("Please ensure NumPy is installed in your environment.")
    raise

class RobotMovements:
    """
    Class for controlling robot movements in Isaac Sim.
    Provides various movement patterns for robot joints.
    """
    
    def __init__(self, robot_articulation, controller=None, max_velocity=0.3):
        """
        Initialize the robot movements controller.
        
        Args:
            robot_articulation: The robot articulation object from Isaac Sim
            controller: Optional articulation controller, if None will use the one from robot_articulation
            max_velocity: Maximum velocity for joint movements (lower values = slower movement)
        """
        self.robot = robot_articulation
        self.controller = controller if controller is not None else robot_articulation.get_articulation_controller()
        self.num_joints = len(self.robot.dof_names)
        self.max_velocity = max_velocity  # Maximum velocity to slow down movements
        
        # Set default joint limits if get_dof_limits is not available
        try:
            # Try to get joint limits using the method if available
            if hasattr(self.robot, 'get_dof_limits'):
                self.joint_lower_limits = self.robot.get_dof_limits()[0]
                self.joint_upper_limits = self.robot.get_dof_limits()[1]
            elif hasattr(self.robot, 'get_joint_limits'):
                # Alternative method name that might exist
                limits = self.robot.get_joint_limits()
                self.joint_lower_limits = limits[0]
                self.joint_upper_limits = limits[1]
            else:
                # Use default limits for Franka Emika robot if methods are not available
                print("Could not find joint limits method, using default Franka limits")
                # Default joint limits for Franka Emika Panda robot (in radians)
                self.joint_lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
                self.joint_upper_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
                
                # If the robot has a different number of joints, adjust the arrays
                if self.num_joints != len(self.joint_lower_limits):
                    print(f"Warning: Default limits are for 7-DOF Franka, but robot has {self.num_joints} joints")
                    # Create arrays of correct size with conservative limits
                    self.joint_lower_limits = np.full(self.num_joints, -1.5)
                    self.joint_upper_limits = np.full(self.num_joints, 1.5)
        except Exception as e:
            print(f"Error getting joint limits: {e}")
            # Fallback to conservative limits
            self.joint_lower_limits = np.full(self.num_joints, -1.5)
            self.joint_upper_limits = np.full(self.num_joints, 1.5)
    
    def get_current_joint_positions(self):
        """
        Helper method to get current joint positions safely.
        
        Returns:
            Current joint positions as numpy array
        """
        try:
            # Try different possible methods to get joint positions
            if hasattr(self.robot, 'get_joint_positions'):
                positions = self.robot.get_joint_positions()
                if isinstance(positions, np.ndarray) and len(positions) == self.num_joints:
                    return positions
            
            if hasattr(self.robot, 'get_dof_positions'):
                positions = self.robot.get_dof_positions()
                if isinstance(positions, np.ndarray) and len(positions) == self.num_joints:
                    return positions
                    
            if hasattr(self.robot, 'joint_positions'):
                positions = self.robot.joint_positions
                if isinstance(positions, np.ndarray) and len(positions) == self.num_joints:
                    return positions
            
            if hasattr(self.robot, 'dof_positions'):
                positions = self.robot.dof_positions
                if isinstance(positions, np.ndarray) and len(positions) == self.num_joints:
                    return positions
                    
            # Fallback: try to get positions from the controller
            if hasattr(self.controller, 'get_joint_positions'):
                positions = self.controller.get_joint_positions()
                if isinstance(positions, np.ndarray) and len(positions) == self.num_joints:
                    return positions
                
            # If all else fails, return zeros
            print("Warning: Could not find method to get joint positions, returning zeros")
            return np.zeros(self.num_joints)
        except Exception as e:
            print(f"Error getting joint positions: {e}")
            return np.zeros(self.num_joints)
        
    def set_random_joint_positions(self, scale_factor=0.5):
        """
        Set random values for each joint within safe limits.
        
        Args:
            scale_factor: Scale factor to limit the range of random values (0.0-1.0)
                          Lower values will keep movements more conservative
        
        Returns:
            The random joint positions that were applied
        """
        try:
            random_positions = np.zeros(self.num_joints)
            
            for i in range(self.num_joints):
                lower = float(self.joint_lower_limits[i])
                upper = float(self.joint_upper_limits[i])
                
                # Scale the range to be more conservative if needed
                range_center = (upper + lower) / 2
                range_half_width = (upper - lower) / 2 * scale_factor
                
                # Generate random value within scaled range
                random_val = np.random.uniform(
                    range_center - range_half_width, 
                    range_center + range_half_width
                )
                
                random_positions[i] = random_val
            
            # Apply the random joint positions with velocity control
            self._apply_position_with_velocity_control(random_positions)
            
            return random_positions
        except Exception as e:
            print(f"Error in set_random_joint_positions: {e}")
            print("If this is a NumPy compatibility error, try running fix_numpy_compatibility.py")
            # Return current positions as fallback
            return self.get_current_joint_positions()
    
    def move_to_home_position(self):
        """
        Move the robot to its home/default position.
        
        Returns:
            The home position joint values
        """
        try:
            # Default home position for Franka (adjust as needed for your robot)
            # Using a slightly better default position for Franka robots instead of all zeros
            if self.num_joints == 7:  # Standard Franka Emika Panda has 7 joints
                # A safe neutral position for Franka
                home_position = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.6, 0.0])
            elif self.num_joints == 9:  # Franka with gripper might have 9 joints
                # A safe neutral position for Franka with gripper
                home_position = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.6, 0.0, 0.04, 0.04])
            else:
                # For any other robot, use zeros but slightly bend the elbows to avoid singularities
                home_position = np.zeros(self.num_joints)
                # Modify elbow joints if there are enough joints
                if self.num_joints >= 3:
                    home_position[1] = -0.3  # Usually the second joint is an elbow
                    if self.num_joints >= 4:
                        home_position[3] = -0.5  # Often the fourth joint is another elbow
            
            # Apply the home position with velocity control
            self._apply_position_with_velocity_control(home_position)
            return home_position
        except Exception as e:
            print(f"Error in move_to_home_position: {e}")
            # If there's an error, try a simpler approach with all zeros
            try:
                simple_home = np.zeros(self.num_joints)
                self._apply_position_with_velocity_control(simple_home)
                return simple_home
            except:
                return self.get_current_joint_positions()
    
    def move_joints_incrementally(self, increments):
        """
        Move joints by specified increments from current position.
        
        Args:
            increments: List or array of increment values for each joint
            
        Returns:
            The new joint positions after applying increments
        """
        try:
            # Get current positions
            current_positions = self.get_current_joint_positions()
            
            # Ensure increments is an array of the right size
            if not isinstance(increments, np.ndarray):
                increments = np.array(increments)
            
            # If sizes don't match, resize increments
            if len(increments) != self.num_joints:
                if len(increments) > self.num_joints:
                    increments = increments[:self.num_joints]  # Truncate
                else:
                    # Extend with zeros
                    temp = np.zeros(self.num_joints)
                    temp[:len(increments)] = increments
                    increments = temp
            
            # Add increments to current positions
            new_positions = current_positions + increments
            
            # Clamp to joint limits
            new_positions = np.clip(
                new_positions, 
                self.joint_lower_limits, 
                self.joint_upper_limits
            )
            
            # Apply the new positions with velocity control
            self._apply_position_with_velocity_control(new_positions)
            return new_positions
        except Exception as e:
            print(f"Error in move_joints_incrementally: {e}")
            return self.get_current_joint_positions()
    
    def _apply_position_with_velocity_control(self, target_positions):
        """
        Apply joint positions with velocity control for smoother, slower movements.
        
        Args:
            target_positions: Target joint positions to move to
            
        Returns:
            None
        """
        try:
            # No need to get current positions just for velocity control
            
            # Ensure target_positions is a numpy array
            if not isinstance(target_positions, np.ndarray):
                target_positions = np.array(target_positions)
                
            # Create velocity array with max velocity for each joint
            velocities = np.full(self.num_joints, self.max_velocity)
            
            # Set joint velocities to limit speed if method exists
            if hasattr(self.controller, 'set_joint_velocities'):
                self.controller.set_joint_velocities(velocities)
            
            # Some controllers support direct velocity limits
            if hasattr(self.controller, 'set_velocity_limits'):
                self.controller.set_velocity_limits(velocities)
            
            # Apply the position target directly
            self.controller.apply_action(target_positions)
            
        except Exception as e:
            print(f"Error in _apply_position_with_velocity_control: {e}")
            # Fallback to direct action without velocity control
            try:
                self.controller.apply_action(target_positions)
            except Exception as e2:
                print(f"Failed to apply fallback action: {e2}")
                # If all else fails, don't try to move the robot
