#!/usr/bin/env python3
"""
Franka Pick and Place Test Runner

This script provides a convenient interface to run different versions
of the Franka pick-and-place scripts for testing and comparison.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from core import setup_isaac_sim_environment

# Available scripts to run
AVAILABLE_SCRIPTS = {
    "original": "franka_pick_place_banana.py",
    "improved": "improved_final_franka_pick_place_banana.py",
    "optimized": "optimized_franka_pick_place.py",
    "advanced": "advanced_franka_pick_place_banana.py",
    "enhanced": "enhanced_franka_pick_place_banana.py",
    "final": "final_franka_pick_place_banana.py"
}

def run_script(script_name):
    """
    Run the specified pick-and-place script
    
    Args:
        script_name: Name of the script to run
    """
    if script_name not in AVAILABLE_SCRIPTS:
        print(f"Error: Unknown script '{script_name}'. Available options are: {', '.join(AVAILABLE_SCRIPTS.keys())}")
        return False
    
    script_path = AVAILABLE_SCRIPTS[script_name]
    print(f"Running {script_name} pick-and-place script: {script_path}")
    
    # Set up the Isaac Sim environment
    setup_isaac_sim_environment()
    
    # Import and run the script
    try:
        script_module = __import__(script_path[:-3])  # Remove .py extension
        if hasattr(script_module, 'main'):
            script_module.main()
        elif hasattr(script_module, 'run_franka_pick_place'):
            script_module.run_franka_pick_place()
        else:
            # Fall back to executing the script directly
            script_full_path = Path(__file__).parent / script_path
            print(f"Running script directly: {script_full_path}")
            exec(open(script_full_path).read())
        
        return True
    except Exception as e:
        print(f"Error running script {script_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point for the script runner"""
    parser = argparse.ArgumentParser(description="Run Franka pick-and-place scripts")
    parser.add_argument("script", nargs="?", default="optimized", 
                        choices=list(AVAILABLE_SCRIPTS.keys()),
                        help="Which pick-and-place script to run")
    
    args = parser.parse_args()
    
    print(f"=== Franka Pick and Place Test Runner ===")
    print(f"Running the '{args.script}' implementation")
    
    success = run_script(args.script)
    
    if success:
        print(f"Script completed successfully!")
    else:
        print(f"Script failed to complete.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
