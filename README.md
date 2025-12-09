# ChestXsim - An open-source framework for realistic chest X-ray tomosynthesis simulations

## 🩻 Overview 
**ChestXsim** is a research-oriented toolkit for simulating **Digital Chest Tomosynthesis (DCT)** from **chest CT** volumes. It provides the full workflow required for DCT simulation. It supports **geometry-aware acquisition**, **physics-based projection models**, and **reconstruction utilities** to obtain paired CT–DCT datasets — ideal for deep learning applications. 

The framework is built around a **pipeline architecture** that allows:
- fully automated end-to-end simularion, or  
- step-wise, customizable execution for flexible experimentation.  

Apart from the source code and materials, this repository also includes a **curated manifest** and helper tools for downloading chest CT volumes from the **Medical Imaging and Data Resource Center (MIDRC)**.

[Main Features](#main-features)  
[Project Structure](#project-structure)  
[Installation](#-installation)  
[MIDRC Data Download](#-midrc-data-download)  
[Usage](#-usage)  
[DL Utilities](#-deep-learning-utilities)  
[Extensibility](#-extensibility)  
[Citation](#-citation)  


## ⚙️ Main Features
- **Modular pipeline architecture** — combine preprocessing, projection, and reconstruction steps for either flexible experimentation or full end-to-end workflows.

- **CT preprocessing** — virtual positioning into DCT geometry, stretcher removal, tissue segmentation, and HU-to-attenuation or density conversion.

- **Geometry-aware acquisition setup** — simulate projections using user-defined tomosynthesis geometries that represent real DCT system configurations.

- **Physics-based projection models** — including beam hardening, inverse-square law effects, detector nonuniformities, and configurable noise models.

- **Reconstruction and interpolation tools** — generate tomosynthesis volumes and align CT and DCT outputs

- **DL model integration** — pretrained models for bed removal and tissue segmentation, with analytical alternatives available.

- **Extensible projection engine wrapper interface** — unified abstraction layer for adding new projection or reconstruction operators and geometries; includes a concrete implementation using the ASTRA Toolbox adapted to DCT geometry.

- **GPU acceleration** — CuPy support for array opearions and CUDA-accelerated projectors through the ASTRA Toolbox.

- **Flexible usage** — accessible via both the Python API and command-line interface.

## 🧱 Project Structure
```text 
ChestXsim/
├── inputs/               # Input CT cases (DICOM or converted volumes)
├── settings/             # Pipeline configuration files (JSON)
├── materials/            # External resources
│   ├── models/           # Pretrained DL models (bed/tissue segmentation)
│   ├── mac/              # Mass attenuation coefficients (MAC) (.mat)
│   ├── spectra/          # X-ray spectra (.mat)
│   └── executables/      # External tools (e.g. projection backends)
├── midrc/                # MIDRC manifest + downloader
├── examples/             # Usage examples and notebooks 
├── src/
│   └── chestxsim/        # Core Python package
│       ├── cli/              # Command-line tools (run_simulation.py, interpolate.py, etc.)
│       ├── core/             # Core abstractions, geometries and pipeline
│       ├── io/               # Readers, writers, path configuration
│       ├── preprocessing/    # Preprocessing steps used in the pipeline (bed removal, unit conversion, etc.)
│       ├── projection/       # Projection steps and physics modules used in the pipeline
│       ├── reconstruction/   # Reconstruction steps (FDK, alignment, etc.)
│       ├── wrappers/         # Operator wrappers (geometry-aware interface + ASTRA-based implementation)
│       └── utility/          # Interpolation, visualization, and helper functions
└── results/             # Output directoty by default 

```

## 📦 Installation 
**ChestXsim** can be installed natively on Windows or run on any platform via Docker.

The framework requires:

- Windows 10/11
- An NVIDIA GPU  
- NVIDIA drivers compatible with **CUDA 12.x**  

Regardless of whether you use local or Docker execution, you must first clone the repository and work from the project root:
```bash
# Clone the repository
git clone https://github.com/ChestXsim-Project.git
cd ChestXsim-Project
``` 

### 1. Install from source (Windows)
> **Note**: ASTRA Toolbox requires compiled binaries that are not available via PyPI.

```bash
# Create environment
conda create -n chestxsim python=3.10
conda activate chestxsim

# Install CUDA-enabled PyTorch (CUDA 12.1)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
conda install conda-forge::astra-toolbox 

# Install package (editable mode)
pip install -e .

# Verify instalation 
python -c "import chestxsim; print('ChestXsim successfully installed')"
python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())"
```

### 2. Build the Docker image
The simplest way to use ChestXsim via Docker is to install Docker Desktop. 
- Download and install from https://docs.docker.com/desktop/
- On Windows, ensure that WSL2 is installed or enabled during setup

> Note: To be able to GPU accelarion inside Docker the host system must have the NVIDIA Container Toolkit. Windows users do not need to install this manually - Docker Desktop + WSL2 provide GPU passthrough automatically. Linux users must install this manually: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

**Verify GPU availability:**
```bash
docker run --rm --gpus all nvidia/cuda:12.2.2-runtime-ubuntu22.04 nvidia-smi
```

**Build the ChestXsim image from the project root:**
```bash
docker build -t chestxsim .
```

## 🩺 MIDRC Data Download

Go to [midrc/](./midrc/) folder. This contains:
- a manifest file (`MIDRC_manifest.json`) listing the collection of chest CT studies suitable for DCT simulation.
- a lightweight graphical downloader that wraps the official **Gen3** client (`gen3-client.exe`),
- documentation and helper files.

To download the data, you will need a valid **MIDRC API Key** (`credentials.json`), which can be generated from your **MIDRC user profile** at the [MIDRC portal](https://data.midrc.org/).


## 🚀 Usage
ChestXsim simulations can be run via **CLI tools** using a JSON configuration file (locally or through Docker).
Users may also import ChestXsim as a Python package and manually construct pipelines for full control over each processing stage.

### 🔧 Before You Start: Core Concepts

Before showing how to run a simulation, here are the core building blocks that ChestXsim uses internally:

- **VolumeData** — the main data container for CT/DCT volumes + metadata.
    ```bash
    VolumeData(
        volume: xp.ndarray,          # 3D or 4D CuPy array
        metadata: MetadataContainer  # voxel size, dims, ID, logs...
    )
    ```
- **Pipeline** — orchestrates a sequence of processing steps  
- **Steps** — small callable objects that transform a `VolumeData` instance. Each step follows the pattern:
    ```bash
    __call__(self, data: VolumeData) -> VolumeData
    ```
    and automatically updated `data.metadata` to mantain a full log of the simulation. 
- **Geometric Operator (Wrapper Interface)** — to execute projection/backprojection operations. 

### 📌 Ways to Use ChestXsim
You can use the toolkit in three ways:  
- [**From configuration files** (easiest)](#1-from-configuration-files)
- [**Manual pipeline construction** (full control)](#2-manual-pipeline-construction)  
- [**Standalone use of ASTRA WRAPPER** (just proj/backproj)](#3-standalone-use-of-astra-wrapper)

#### 1. From configuration files
A minimal configuration for simulating a VolumeRAD-like DCT system at 120 kVp:
```jsonc
{
  "modality": "DCT",                  // Simulation modality: Digital Chest Tomosynthesis

  "geometry": {                       // Acquisition geometry (represents the DCT system)
    "detector_size": [4288, 4288],    // Detector width × height (pixels)
    "pixel_size": [0.1, 0.1],         // Pixel size (mm)
    "SDD": 1800.0,                    // Source-to-detector distance (mm)
    "bucky": 14.47,                   // Bucky distance (mm)
    "step_mm": 16.077,                // Source step size (mm)
    "nprojs": 61                      // Number of projections
  },

    "preprocessing": {                // CT preprocessing steps to obtain density volumes 
            "bed_removal": {"threshold": -200, "save_mask": false, "save": false},
            "air_cropping": {"axis": 1, "tol": 5, "delta": 3, "channel": 0, "save": false},
            "volume_extension": {"target_height": 600.0, "chest_center": 150, "save": false},
            "tissue_segmentation": {"threshold": 300, "tissue_types": ["bone", "soft"], "save": false},
            "unit_conversion": {"units": "density", "tissue_types": ["bone", "soft"], "save": true}
        },

  "projection": {                     // Simple X-ray projection simulation
    "projection": {"opt": "astra", "channel_wise": true, "save": true},
    "physics_effect": {
      "voltage": 120,                 // Tube voltage (kVp)
      "poly_flag": true,              // Enable polychromatic spectrum
      "save": true
    }
  },

  "reconstruction": {                 // Tomosynthesis reconstruction
    "FDK": {
      "opt": "astra",                 // Use the same ASTRA-based operator
      "match_input": true,            // Match original volume dimensions
      "save": true
    }
  }
}

``` 

**Run via Python API**
```bash
from chestxsim.pipeline import build_pipeline
from chestxsim.io.readers import DicomReader
import json

# Load configuration file
with open(".\settings\volumeRAD_poly_FDK.json") as f:
    cfg = json.load(f)

# Load CT volume (DICOM → volumeData)
ct = DicomReader(convert_to_HU=True, clip_values=[-1000, 3000]).read(
    "inputs/Case_001/"
)

# Build the full simulation pipeline from config
pipe = build_pipeline(cfg, mode = None, output_folder="results")

# Execute all steps: preprocessing → projection → reconstruction
dct_out = pipe.execute(ct)
```
The pipeline builder supports *partial execution* via the `mode` flag.  
This lets you reuse previous results:  
> - `mode=0` → preprocessing only  
> - `mode=1` → projection only  
> - `mode=2` → reconstruction only  

**Run via CLI (local installation)**  
General syntax:
```bash 
run_simulation --input PATH_TO_CT_STUDIES --config settings/CONFIG.json [--mode 0|1|2] [--output OUTPUT_DIR]
```
To run the full simulation:  
```bash
run_simulation -i .\inputs -c .\settings\volumeRAD_poly_FDK.json -o .\results\volumeRAD_poly_FDK
```
Run a single module by setting the ``--mode`` flag.
For example, generate projections from previously computed density volumes:
```bash 
run_simulation -i .\results\volumeRAD_poly_FDK\CT_converted\density -c .\settings\volumeRAD_poly_FDK.json -o .\results\volumeRAD_poly_FDK -m 1
```

**Run via CLI inside Docker**
General syntax:
```bash 
docker run --rm --gpus all -v <PROJECT_ROOT>:/app -e CHESTXSIM_ROOT=/app chestxsim run_simulation [arguments]
```

Run full simulation (Equivalent to the local example above)
```bash 
docker run --rm --gpus all \
  -v $(pwd):/app \
  -e CHESTXSIM_ROOT=/app \
  chestxsim \
  run_simulation \
  -i /app/inputs \
  -c /app/settings/volumeRAD_poly_FDK.json \
  -o ./results/volumeRAD_poly_FDK

```
Run a single module, (Equivalent to the local example above)
docker run --rm --gpus all \
  -v $(pwd):/app \
  -e CHESTXSIM_ROOT=/app \
  chestxsim \
  run_simulation \
  -i /app/results/volumeRAD_poly_FDK/CT_converted/density \
  -c /app/settings/volumeRAD_poly_FDK.json \
  -o /app/results/volumeRAD_poly_FDK \
  -m 1

#### 2. Manual pipeline construction
You can manually compose a pipeline using `.add(step, save=...)`. For example, to create a preprocessing pipeline:
```bash 
from chestxsim.pipeline import Pipeline
from chestxsim.preprocessing.steps import *

# Create pipeline and add steps
pipe = Pipeline()
pipe.add(BedRemover(threshold=-200))
pipe.add(AirCropper(axis=1))
pipe.add(VolumeExtender(ext_vals_mm=[100, 100]))
pipe.add(TissueSegmenter(threshold=300, tissue_types=["bone","soft"]))
pipe.add(UnitConverter(units="density", tissue_types=["bone","soft"]), save=True)

# Run the pipeline
processed = pipe.execute(ct_data)
```

#### 3. Standalone use of ASTRA WRAPPER
ChestXsim provides a **geometry-aware operator wrapper** specific for the ASTRA Toolbox that can be used independently of the simulation pipeline. This means you can run forward and backprojections directly on CuPy volumes. 

**👉 ASTRA_Tomo (DCT-specific ASTRA wrapper)**
```bash
from chestxsim.core.geometries import TomoGeometry
from chestxsim.wrappers.astra import ASTRA_Tomo

# Define DCT geometry
geo = TomoGeometry(
    detector_size=(4288, 4288),
    pixel_size=(0.1, 0.1),
    binning_proj=8,
    SDD=1800.0,
    bucky=14.47,
    step_mm=16.077,
    nprojs=60
)

# Create ASTRA operator
opt = ASTRA_Tomo(geometry=geo)

# Example input parameters
vx_xyz = (0.84, 0.84, 1.25)                # Input voxel size (mm)
reco_dim_xyz = (400, 80, 200)              # Desired reconstruction volume dimensions (px)
reco_vx_xyz   = (1.25, 5.00, 1.25)         # Desired tomosynthesis voxel size (mm)

# Forward and backward projection
projs = opt.project(input_volume, vx_xyz)
recon = opt.backproject(projs, reco_dim_xyz, reco_vx_xyz)
```
Internally, ASTRA_Tomo configures ASTRA’s 3D cone-beam geometry and dispatches CUDA-accelerated routines:
- **Forward projection** → **'FP3D_CUDA'**
- **Backprojection → 'BP3D_CUDA**'

In constrast, when used inside ChestXsim pipeline, the same operator is automatically provided to steps such as `projection()`, `FDK()`, and `SART()`. ChestXsim automatically:
- pases voxel size, detector geometry, and physical spacing 
- manages volume sahpe and coordinate conventions
based on information stored in each ```VolumeData.metadata``` object. 
When reconstruction is run with  `match_input=True`, the resulting tomosynthesis volume preserves the original CT dimensions in millimeters, while adopting the desired DCT voxel spacing, typically (1.25, 5.00, 1.25).

### 4. Guides and Notebooks 
See the full tutorials:
- [CLI Cheatsheet]
- [Core Structures](./examples/0.%20Data_containers.ipynb)  
- [I/O Management](./examples/1.%20CT%20Readers.ipynb)
- [Pipeline](./examples/5.%20Pipeline.ipynb)
- [ASTRA Wrapper](./examples/6.%20Astra%20Wrapper.ipynb)  
- [Preprocessing Module](./examples/2.%20Preprocessing.ipynb)  
- [Projection Module](./examples/3.%20Projection.ipynb)
- [Reconstruction Module](./exampless/4.%20Preprocessing.ipynb)


## 🧠 Deep Learning Utilities
**Available pretrained models**
- **Bed Removal Model** — segments non-patient structures (stretcher/table) for removal.  
  Weights: `materials/models/model_BedSeg.pt`

- **Bone Segmentation Model** — generates a bone mask to separate bone from soft tissue.  
  Weights: `materials/models/model_BoneSeg.pt`

**Implementation notes**
- Both models use a U-Net architecture with a ResNet encoder (PyTorch).
- Inference is performed slice-wise with batched execution.
- DLPack enables efficient CuPy ↔ PyTorch data transfer for GPU inference.

## 🔧 Extensibility
ChestXsim was designed for **flexible and extensible** use. 
- Add new processing steps and register them to the pipeline. 
- Extend projection engines via the wrapper interface. 
- Integrate new physics effects or materials defenitions. 
- Integrate new geometry types. 

## 📝 Related Publications 

[Hermosilla, B., Lorente-Mur, A., Del Cerro, C. F., Desco, M., & Abella, M. (2025, June). ChestXsim: An Open-Source Framework for Realistic Chest X-Ray Tomosynthesis Simulations. In 2025 IEEE 38th International Symposium on Computer-Based Medical Systems (CBMS) (pp. 123-124).](IEEE.https://ieeexplore.ieee.org/abstract/document/11058874)

[Del Cerro, C. F., Galán, A., García Blas, J., Desco, M., & Abella, M. (2022, October). New reconstruction methodology for chest tomosynthesis based on deep learning. Proceedings of SPIE, 12304, 7th International Conference on Image Formation in X-Ray Computed Tomography, 123042X](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12304/2646600/New-reconstruction-methodology-for-chest-tomosynthesis-based-on-deep-learning/10.1117/12.2646600.full)
