#!/usr/bin/env python3
"""
List all objects in the USD scene to identify the banana and tray objects
"""

import time
from pathlib import Path
from core import setup_isaac_sim_environment

THIS_FOLDER = Path(__file__).parent.resolve()
USD_SCENE_PATH = str(THIS_FOLDER / "scenes" / "franka_blocks_manual.usd")

def list_scene_objects():
    """
    Load the USD scene and list all objects
    """
    # Import SimulationApp first to initialize the simulation
    from isaacsim import SimulationApp

    # Create the simulation app
    simulation_app = SimulationApp({"headless": True})
    
    # Import necessary modules after SimulationApp is initialized
    import omni.usd
    
    print(f"Opening scene: {USD_SCENE_PATH}")
    usd_context = omni.usd.get_context()
    usd_context.open_stage(USD_SCENE_PATH)
    simulation_app.update()
    
    # Get the stage after opening it
    stage = usd_context.get_stage()
    
    # List all objects in the scene
    print("\nScene objects:")
    for prim in stage.Traverse():
        path = prim.GetPath()
        typeName = prim.GetTypeName()
        print(f"- {path} (Type: {typeName})")
    
    # Close the simulation
    simulation_app.close()

if __name__ == "__main__":
    # Set up the environment and run the scene exploration
    setup_isaac_sim_environment()
    list_scene_objects()
