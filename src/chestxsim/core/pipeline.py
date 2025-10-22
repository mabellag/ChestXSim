"""
Pipeline engine for ChestXsim.

This module enables the creation and execution of modular simulation pipelines
for Digital Chest Tomosynthesis. Pipelines can be constructed using a configuration
dictionary in combination with `PROCESSING_STEP_REGISTRY`, or manually using
the `add()` method to include callable steps.

It operates on `volumeData` objects and supports geometry-aware operators with
dynamic assignment of projection and reconstruction kernels (ASTRA, FuXSim, Raptor).
"""


import gc
import inspect
from typing import Union, Optional, List, Callable, Any

from chestxsim.core.device import xp
from chestxsim.io.save_manager import SaveManager
from chestxsim.io.paths import EXECUTABLES_DIR
from chestxsim.core.data_containers import volumeData, SourceSpectrum, MACRepo
from chestxsim.core.geometries import Geometry, TomoGeometry, CBCTGeometry

from chestxsim.preprocessing.steps import *
from chestxsim.projection.steps import *
# from chestxsim.reconstruction.steps import *
from chestxsim.wrappers import *
# from chestxsim.utility.interpolation import Interpolator



# Maps configuration keywords to corresponding processing step classe
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
    # "backprojection": BackProjector,
    # "FDK": FDK, 
    # "FDK-Fuxsim": FuximFDK,
    # "RaptorFDK": RaptorFDK, 
    # "CT_alignment_to_reco": Interpolator



}

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

    def add(self, step: Callable, save: Optional[bool] = False):
        """
        Add a callable processing step to the pipeline.
        """
        if callable(step):
            self.steps.append((step, save))
        else:
            raise ValueError("Step must be callable")


    def execute(self, input: volumeData):
        """
        Execute all registered steps sequentially on the input volumeData.
        Optionally saves intermediate results and clears memory if using CuPy.
        """

        processed_data = input

        for step, save in self.steps:
            processed_data = step(processed_data)
            print(f"{step.__class__.__name__} done")

            if save:
                self.save_manager.save_step(step.__class__.__name__, processed_data)

            # Force free GPU memory (if using CuPy)
            if xp.__name__ == "cupy":
                xp.get_default_memory_pool().free_all_blocks()

            # print(processed_data.volume.shape)

            gc.collect()  

        return processed_data


    def add_step_from_config(self, config: dict):
        """
        Add pipeline steps from a configuration dictionary.
        Handles geometry and kernel injection if required.
        """
        for step_name, step_config in config.items():
            if not step_config.get("enabled", False):
                continue
        
            cls = PROCESSING_STEP_REGISTRY.get(step_name)
            if not cls:
                print(f"Warning: No registered step found for '{step_name}'")
                continue

            params = self._filter_params(cls, step_config)
         
            # Inject geometry if needed
            if "geometry" in inspect.signature(cls.__init__).parameters:
                use_geom_flag = step_config.get("useGeometry", False)
                if params.get("geometry") is None and use_geom_flag:
                    params["geometry"] = self.geometry
    
            # Projection or reconstruction step needing a kernel
            if step_name == "projection" or step_name.lower().startswith("fdk"):
                if "fuxsim" in step_name.lower():
                    kernel_name = "fuxsim"
                elif "raptor" in step_name.lower():
                    kernel_name = "raptor"
                else:
                    kernel_name = step_config.get("opt", step_config.get("kernel", "astra")).lower()
                
                # print(self.kernel_cache)
                if kernel_name not in self.kernel_cache:
                    self.kernel_cache[kernel_name] = create_operator(self.geometry, kernel_name)
                    print("operator created")

                params["opt"] = self.kernel_cache[kernel_name]
                print(params["opt"].DOD)
                # print(self.kernel_cache)

            # if step_name == "spectrum_effect":
            #     params["physics"] = Physics.from_dict(step_config.get("physics"))

            save_flag = step_config.get("save", False)
            self.add(cls(**params), save=save_flag)

        return self

    @staticmethod
    def _filter_params(cls, config_params: dict) -> dict:
        """
        Filters configuration keys to match constructor arguments for a given class.
        Passes through additional keys if the class accepts **kwargs.
        """
        sig = inspect.signature(cls.__init__)
        allowed_keys = set(sig.parameters) - {"self"}
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

        filtered = {k: v for k, v in config_params.items() if k in allowed_keys}

        if accepts_kwargs:
            extra_kwargs = {
                k: v for k, v in config_params.items()
                if k not in allowed_keys and k != "enabled"
            }
            filtered.update(extra_kwargs)

        return filtered


def build_pipeline(config: dict, mode: Optional[int]= None, output_folder: Optional[int]= None) -> Pipeline:
    """
    Builds a pipeline from configuration dictionary.

    Args:
        config (dict): Full pipeline config including geometry and modules.
        mode (int, optional): One of {0: preprocessing, 1: projection, 2: reconstruction}.
        output_folder (str, optional): Output directory for saved steps.

    Returns:
        Pipeline: Configured pipeline instance.
    """
    pipeline = Pipeline(base_save_dir=output_folder)
    geometry_dict = config.get("geometry")
    modality = config.get("modality", "CBCT").upper()
    print(modality)

    pipeline.geometry = CBCTGeometry.from_dict(geometry_dict) if modality == "CBCT" else TomoGeometry.from_dict(geometry_dict)
    print(pipeline.geometry)


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
            # print(config.get(section, {}))
            pipeline.add_step_from_config(config.get(section, {}))
        else:
            raise ValueError(f"Invalid mode '{mode}'. Expected 0 (preprocessing), 1 (projection), or 2 (reconstruction).")

    return pipeline


# def create_operator(
#     geometry_obj: Geometry,
#     kernel_name: str,
#     path_to_executable: Optional[str] = None,    
#     ) -> Astra_OP:
#     """
#     Factory function to create a projection or reconstruction operator.

#     Args:
#         geometry_obj (Geometry): Geometry instance (CBCT or Tomo).
#         kernel_name (str): 'astra', 'fuxsim', or 'raptor'.
#         path_to_executable (str, optional): Override path for external .exe tools.

#     Returns:
#         Configured operator instance.
#     """
#     modality = geometry_obj.modality

#     kernel = kernel_name.lower()

#     if kernel == "astra":
#         return Astra_CBCT(geometry_obj) if modality == "CBCT" else Astra_Tomo(geometry_obj)

#     # elif kernel == "fuxsim":
#     #     if not path_to_executable:
#     #         path_to_executable = get_path_executable("fuxsim")
#     #     if not path_to_executable:
#     #         raise ValueError("FuXSim requires a valid 'path_to_executable'.")
#     #     return FuXSim_CBCT(path_to_executable, geometry_obj) if modality == "CBCT" else FuXSim_Tomo(path_to_executable, geometry_obj)

#     # elif kernel == "raptor":
#     #     if not path_to_executable:
#     #         path_to_executable = get_path_executable("raptor")
#     #     if not path_to_executable:
#     #         raise ValueError("Raptor requires a valid 'path_to_executable'.")
#     #     return Raptor_OP(path_to_executable, geometry=geometry_obj)

#     else:
#         raise ValueError(f"Unsupported kernel type: '{kernel_name}'")


# def get_path_executable(program_name:str):
#     program_dir = EXECUTABLES_DIR / program_name
#     if not program_dir.exists():
#         raise FileNotFoundError(f"Executable directory not found: {program_dir}")

#     exe_files = [f for f in program_dir.glob("*.exe")]
#     if not exe_files:
#         raise FileNotFoundError(f"No .exe file found in: {program_dir}")

#     return str(exe_files[0])

def create_operator():
    pass 