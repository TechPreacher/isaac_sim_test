import os
import platform
import shutil


def setup_isaac_sim_environment():
    """
    Set up the environment variables needed for Isaac Sim based on the batch file setup.
    This includes setting CARB_APP_PATH, ISAAC_PATH, and EXP_PATH.
    """
    # Determine the Isaac Sim installation path
    # In a real implementation, you might need to adjust this based on your system
    isaac_sim_path = os.environ.get("ISAAC_SIM_PATH", "c:/dev/tools/isaac-sim")

    # Set environment variables similar to the batch file
    os.environ["CARB_APP_PATH"] = os.path.join(isaac_sim_path, "kit")
    os.environ["ISAAC_PATH"] = isaac_sim_path
    os.environ["EXP_PATH"] = os.path.join(isaac_sim_path, "apps")
    os.environ["ACCEPT_EULA"] = "Y"
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"

    # Python executable handling
    if "PYTHONEXE" not in os.environ:
        python_path = os.path.join(isaac_sim_path, "kit", "python")
        kit_exe = os.path.join(python_path, "kit.exe")
        python_exe = os.path.join(python_path, "python.exe")

        # Create kit.exe if it doesn't exist (similar to batch file)
        if (
            platform.system() == "Windows"
            and not os.path.exists(kit_exe)
            and os.path.exists(python_exe)
        ):
            try:
                shutil.copy(python_exe, kit_exe)
                print(f"Created {kit_exe} from {python_exe}")
            except Exception as e:
                print(f"Warning: Failed to create kit.exe: {e}")

        # Set PYTHONEXE environment variable
        os.environ["PYTHONEXE"] = kit_exe if os.path.exists(kit_exe) else python_exe

    print(f"Environment setup complete. ISAAC_PATH: {os.environ.get('ISAAC_PATH')}")
