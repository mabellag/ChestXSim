from pathlib import Path 

# All paths below are resolved relative to the  location of this file.
# This ensures that paths remain valid regardless of the current working directory,
# allowing the codebase and individual modules to be used independently or as part of a pipeline.
# Assumes this file is located at: chestXsim-project/src/chestxsim/io/paths.py

# Dynamically find project root 
def find_project_root(marker: str = "materials") -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find '{marker}' folder in any parent directory of {__file__}")


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
    "Interpolator": "CT_interpolated"
    
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

