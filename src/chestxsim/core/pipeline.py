"""
Pipeline engine for ChestXsim.

This provides the core execution framework that drives all simulation
workflows in ChestXsim, including preprocessing, projection, physics modelling,
and reconstruction. A `Pipeline` is a lightweight orchestrator that applies a
sequence of modular processing steps—each operating on a `volumeData` object—
and produces intermediate and final results in a reproducible manner.

Key Features
------------
• **Modular architecture**  
  Each operation (bed removal, tissue segmentation, conversion to μ/density,
  projection, noise modelling, FDK, etc.) is implemented as a small callable
  class. The pipeline simply executes them sequentially.

• **Configuration-driven execution**  
  Pipelines can be constructed directly from a user-provided configuration
  dictionary 

• **Geometry-aware processing**  
  Many steps require acquisition geometry (Tomo or CBCT). The pipeline injects
  the correct `Geometry` instance automatically when needed.

• **Dynamic operator creation**  
  Projection and reconstruction kernels are created on
  demand and cached for reuse. This allows switching backend operators at
  runtime without modifying step implementations.

• **Automatic metadata propagation**  
  Each step updates the `volumeData.metadata` object, enabling full provenance
  tracking and reproducibility of simulation workflows.

• **Optional saving of intermediate states**  
  Every step can persist its output through the `SaveManager`

Usage
-----
Pipelines can be created manually:

    p = Pipeline()
    p.add(...)
    p.add(...)
    out = p.execute(input_volume)

or automatically from a configuration file:

    pipeline = build_pipeline(config)
    output = pipeline.execute(ct_data)

Structure
---------
• `PROCESSING_STEP_REGISTRY`  
  Maps string identifiers from config files to processing step classes.

• `Pipeline`  
  - Holds the list of steps  
  - Injects geometry/operators into steps  
  - Runs the pipeline  
  - Manages GPU memory and intermediate saves  

• `build_pipeline`  
  High-level entry point that:
    - Creates geometry from the config  
    - Instantiates the pipeline  
    - Populates it with preprocessing, projection, and/or reconstruction steps  

"""

import gc
import inspect
from typing import Union, Optional, List, Callable, Any

from chestxsim.core.device import xp
from chestxsim.io import SaveManager
from chestxsim.core.geometries import *
from chestxsim.core.data_containers import *
from chestxsim.preprocessing.steps import *
from chestxsim.projection.steps import *
from chestxsim.reconstruction.steps import *
from chestxsim.wrappers import *
from chestxsim.utility.interpolation import Interpolator



#--- KEYWORDS MAPPING FOR STEP REGISTRY --------------
PROCESSING_STEP_REGISTRY = {
    "bed_removal": BedRemover,
    "air_cropping": AirCropper,
    "unit_conversion": UnitConverter,
    "volume_extension": VolumeExtender,
    "volume_flipping": VolumeFlipper,
    "tissue_segmentation": TissueSegmenter,
    "projection": Projection,
    "Physics_effect": PhysicsEffect,
    "noise_effect": NoiseEffect,
    "backprojection": BackProjector,
    "FDK": FDK, 
    "CT_resampled": Interpolator,
    "CT_rotated": VolumeRotate, 

}

# --- PIPELINE CLASS --------------------------------------------------------
class Pipeline:
    """
    Modular simulation pipeline that applies a sequence of processing steps
    to a `volumeData` object. Steps can be added manually or via a config.
    """
    def __init__(self, steps: Optional[List[Callable]] = None, base_save_dir: str = "results"):
        self.steps = steps if steps is not None else []
        self.save_manager = SaveManager(base_save_dir)
        self.geometry: Optional[Geometry] = None
        self.kernel_cache = {}

    def _log(self, msg:str):
        print(f"[Pipeline] {msg}")

    def add(self, step: Callable, save: Optional[bool] = False):
        """
        Add a callable processing step to the pipeline.
        """
        if callable(step):
            self.steps.append((step, save))
        else:
            raise ValueError("[Pipeline] Step must be callable")


    def execute(self, input: volumeData):
        """
        Execute all registered steps sequentially on the input volumeData.
        Optionally saves intermediate results and clears memory if using CuPy.
        """
        self._log("Pipeline execution started.")
        self._log(f"Number of steps: {len(self.steps)}")

        processed_data = input

        for i, (step, save) in enumerate(self.steps):
            step_name = step.__class__.__name__
            self._log(f"({i}/{len(self.steps)}) {step_name} — running")
            processed_data = step(processed_data)
            self._log(f"{i}/{len(self.steps)}) {step_name} — done")
            if save:
                self.save_manager.save_step(step.__class__.__name__, processed_data)

            # Force free GPU memory (if using CuPy)
            if xp.__name__ == "cupy":
                xp.get_default_memory_pool().free_all_blocks()

            # print(processed_data.volume.shape)
            gc.collect()  
        
        self._log("Pipeline execution finished.")
        return processed_data


    def add_step_from_config(self, config: dict):
        """
        Add pipeline steps from a configuration dictionary.
        Handles geometry and kernel injection if required.
        """
        for step_name, step_config in config.items():
            cls = PROCESSING_STEP_REGISTRY.get(step_name)
            if not cls:
                self._log(f"No registered step found for '{step_name}', skipping.")
                continue

            sig = inspect.signature(cls.__init__)
            param_names = set(sig.parameters) - {"self"}
            params = {k: v for k, v in (step_config or {}).items() if k in param_names}
            
            # inject geometry from pipeline if steps requires it 
            if "geometry" in param_names and "geometry" not in params:
                params["geometry"] = self.geometry

            # inject operator required for projection/backprojection steps
            if "opt" in param_names: 
                kernel_name =  step_config.get("opt", step_config.get("kernel", "astra")).lower()
                if kernel_name not in self.kernel_cache:
                     self.kernel_cache[kernel_name] = create_operator(kernel_name, self.geometry )
                     self._log(f"Initialized operator '{kernel_name}'.")          
                params["opt"] = self.kernel_cache[kernel_name]

                        
            # inject source if setp requires spectrum 
            if "source" in param_names and "source" not in params:
                params["source"] = SourceSpectrum(
                    I0 = step_config.get("I0", 1e3),
                    voltage = step_config.get("voltage", 120),
                    poly_flag = step_config.get("poly_flag", False),
                     effective_energy  = step_config.get("effective_energy", None),                                                                
                )
                source = "polychromatic" if step_config.get("poly_flag") else "monochromatic"
                self._log(f'Initialized {source} X-ray source at {step_config.get("voltage")}')

            save_flag = step_config.get("save", False)
            self.add(cls(**params), save=save_flag)
            self._log(f"Added step: {cls.__name__} (save={save_flag})")

        # self._log("Pipeline configuration completed.")
        return self


# --- PIPELINE FACTORY / HELPERS 
def build_pipeline(config: dict, mode: Optional[int]= None, output_folder: Optional[int]= None) -> Pipeline:
    """
    Builds a pipeline from configuration dictionary.
    It uses geometry injection from config dict 

    Args:
        config (dict): Full pipeline config including geometry and modules.
        mode (int, optional): One of {0: preprocessing, 1: projection, 2: reconstruction}.
        output_folder (str, optional): Output directory for saved steps.

    Returns:
        Pipeline: Configured pipeline instance.
    """

    print("[Pipeline] Building simulation pipeline...")
    geom_cfg  = config.get("geometry")
    geom_id  = config.get("modality", "CBCT").upper()
    geometry = create_geometry_from_id(geom_id, **geom_cfg)
    print(f"[Pipeline] Geometry initialized ({geom_id}) with {geometry.nprojs} projections.")

    pipeline = Pipeline(base_save_dir=output_folder)
    pipeline.geometry = geometry

    module_mapping = {
        0: "preprocessing",
        1: "projection",
        2: "reconstruction",
    }

    if mode is None:
        for module in module_mapping.values():
            pipeline.add_step_from_config(config.get(module, {}))
    
    else:
        section = module_mapping.get(mode)
        if section:
            print(f"[Pipeline] Initializing pipeline for {section}")
            pipeline.add_step_from_config(config.get(section, {}))
        else:
            raise ValueError(f"[Pipeline] Invalid mode '{mode}'. Expected 0 (preprocessing), 1 (projection), or 2 (reconstruction).")

    print("[Pipeline] Pipeline configuration completed.")
    return pipeline


