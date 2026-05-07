"""
Colab runtime setup for Lecture 1.

The lecture's interactive cells use only numpy and matplotlib for the live
demonstrations — both come pre-installed on Colab and in any standard
scientific Python environment, so no installs are typically required.

The functions below are convenience helpers if you want to verify the
environment, set deterministic random state, or pre-stage any data needed
for live demos.

Usage on Colab:
    # Most cells will Just Work. If you want to pin versions defensively:
    !pip install --quiet "numpy>=1.24,<2.0" "matplotlib>=3.7,<4.0"

Usage locally:
    A conda environment is provided at workshop_root/env/environment.yml.
    The lecture's interactive cells run with the same environment.
"""

# Pinned versions known to work with the lecture's interactive cells.
# These are deliberately conservative — anything within the pinned ranges
# should display the demo plots correctly.
PINNED_VERSIONS = {
    "python": ">=3.10,<3.13",
    "numpy": ">=1.24,<2.0",
    "matplotlib": ">=3.7,<4.0",
}


def install_minimal(verbose: bool = True) -> None:
    """Install the minimal dependencies the lecture's demo cells need.

    Idempotent — safe to call multiple times. On Colab, uses pip with --quiet.
    Locally, prints a hint to use the conda environment instead.
    """
    import sys
    in_colab = "google.colab" in sys.modules

    if in_colab:
        import subprocess
        cmd = [
            sys.executable, "-m", "pip", "install", "--quiet",
            "numpy>=1.24,<2.0",
            "matplotlib>=3.7,<4.0",
        ]
        subprocess.check_call(cmd)
        if verbose:
            print("Installed: numpy, matplotlib (pinned versions)")
    else:
        if verbose:
            print("Local environment: use the conda env at env/environment.yml.")
            print("If you need ad-hoc install, run:")
            print("  pip install 'numpy>=1.24,<2.0' 'matplotlib>=3.7,<4.0'")


def verify_environment() -> dict:
    """Return a dict describing the runtime environment.

    Used by the first interactive cell to confirm the notebook is happy
    before running the demos.
    """
    import sys
    import platform

    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "in_colab": "google.colab" in sys.modules,
        "modules": {},
    }

    for mod_name in ("numpy", "matplotlib"):
        try:
            mod = __import__(mod_name)
            info["modules"][mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info["modules"][mod_name] = None

    return info


def set_random_state(seed: int = 42) -> None:
    """Set deterministic random state for reproducible demos.

    The lecture's demo cells already set seeds inline, but this helper
    centralizes it for any custom additions.
    """
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


# When run as a script, print the environment summary.
if __name__ == "__main__":
    info = verify_environment()
    print("Lecture 1 — Colab/local runtime check")
    print(f"  Python    : {info['python_version']}")
    print(f"  Platform  : {info['platform']}")
    print(f"  Colab     : {info['in_colab']}")
    print(f"  numpy     : {info['modules']['numpy']}")
    print(f"  matplotlib: {info['modules']['matplotlib']}")
