from pathlib import Path 
import os 

# All paths below are resolved relative to the  location of this file.
# This ensures that paths remain valid regardless of the current working directory,

_MARKERS = ("pyproject.toml", ".git")

def find_project_root() -> Path:
    env = os.getenv("CHESTXSIM_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
        raise RuntimeError(f"CHESTXSIM_ROOT points to a non-existing path: {p}")

    # auto-detect by walking up from this file
    current = Path(__file__).resolve()
    for parent in current.parents:
        if any((parent / m).exists() for m in _MARKERS):
            return parent

    raise RuntimeError(
        "Cannot find project root. Set CHESTXSIM_ROOT manually, e.g.:\n"
        "  setx CHESTXSIM_ROOT \"D:\\bhermosi\\chestxsim-project\"  (Windows)\n"
        "  export CHESTXSIM_ROOT=/path/to/chestxsim-project  (Linux/macOS)"
    )

PROJECT_ROOT = find_project_root()

# Materials 
MATERIALS_DIR = PROJECT_ROOT / "materials"
EXECUTABLES_DIR = MATERIALS_DIR / "executables"
MODELS_DIR = MATERIALS_DIR / "models"
MAC_DIR = MATERIALS_DIR / "mac"
SPECTRUM_DIR = MATERIALS_DIR / "spectra"

# Other
RESULTS_DIR = PROJECT_ROOT / "results"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
INPUTS_DIR = PROJECT_ROOT / "inputs"
SETTINGS_DIR = PROJECT_ROOT / "settings"

# Maping steps in pipeline to folder for saving 
STEP_TO_FOLDER = {
    "BedRemover": "CT_without_bed",
    "AirCropper": "CT_air_cropped", 
    "VolumeExtender": "CT_extended",
    "VolumeFlipper": "CT_flipped",
    "TissueSegmenter": "CT_segmented", 
    "UnitConverter": "CT_converted",
    "Projection": "Projections",
    "PhysicsEffect": "EnergyProjections", 
    "NoiseEffect": "NoisyProjections",
    "RaptorFDK": "RaptorFDK",
    "FuximFDK": "FuximFDK",
    "FDK": "FDK",
    "SART": "SART",
    "Interpolator": "CT_interpolated",
    "VolumeRotate":"CT_rotated"
}

UNITS_TO_FOLDER = {
    "mu": "mus",
    "density": "density"
}

TISSUE_TO_FOLDER = {
    "bone": "bone",
    "soft": "soft_tissue",
 
}

SPECTRUM_TO_FOLDER = {
    "polychromatic": "polychromatic",
    "monochromatic": "monochromatic"
}

