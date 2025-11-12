"""
Factory to create a geometric operator wrapper for a given projection engine.

Supported engines:
    - ASTRA   → uses GPU by default
    - FuXSim  → requires executable path
    - Raptor  → requires executable path
"""

from typing import Optional, Dict, Type
import importlib
import pathlib

from chestxsim.core.geometries import Geometry
from chestxsim.wrappers.base import GeometricOp, OpMod, opmod_from_geometry
from typing import Optional

def _load_wrapper(engine: str):
    """Lazy-load the selected engine module and return its MODALITY_REGISTRY"""
    k = engine.strip().lower()
    module_map = {
        "astra":  "chestxsim.wrappers.astra",
        "fuxsim": "chestxsim.wrappers.fuxsim",
        "raptor": "chestxsim.wrappers.raptor",
    }

    modname = module_map.get(k)
    if not modname:
        raise ValueError(f"Unsupported engine '{engine}'. Valid options: {', '.join(module_map)}")

    mod = importlib.import_module(modname)

    # Required: each module must define MODALITY_REGISTRY
    try:
        registry: Dict[OpMod, Type[GeometricOp]] = getattr(mod, "MODALITY_REGISTRY")
    except AttributeError:
        raise RuntimeError(f"{modname} must define MODALITY_REGISTRY: Dict[OpMod, Type[GeometricOp]].")
    
    return registry

def create_operator(
    engine: str,
    geometry: Geometry,
    path_to_executable: Optional[str] = None
) -> GeometricOp:
    """
    Create an operator instance corresponding to a given geometry and projection engine.

    Args:
        engine: one of {"astra", "fuxsim", "raptor"...}
        geometry: Geometry instance describing acquisition setup
        path_to_executable: optional path to external engine executable (for FuXSim/Raptor)
    """
    opmod = opmod_from_geometry(geometry)
    registry = _load_wrapper(engine)

    # Retrieve operator class from engine’s registry
    try:
        op_class = registry[opmod]
    except KeyError:
        raise ValueError(f"Engine '{engine}' does not support modality '{opmod.value}'")

    k = engine.strip().lower()

    if k == "astra":
        return op_class(geometry=geometry)

    # --- External engines (FuXSim / Raptor) ---
    if k in {"fuxsim", "raptor"}:
        if not path_to_executable:
            path_to_executable = get_path_executable(k)
        return op_class(geometry=geometry, executable=path_to_executable)

    raise ValueError(f"Unsupported engine '{engine}'")


    
def get_path_executable():
    pass 






