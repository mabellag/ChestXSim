"""
Base abstractions for geometric operators wrapping external projection engines.

A geometric operator is a thin backend-agnostic interface (ASTRA, FuXSim, Raptor, …)
that knows how to:
    - project a volume into a sinogram,
    - backproject a sinogram into a volume,
    - run reconstruction algorithm from the own software if available (e.g. FBP/FDK, iterative methods)
for a given acquisition `Geometry`.
"""

from chestxsim.core.geometries import Geometry, Modality, BeamGeom
from chestxsim.core.device import xp
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from enum import Enum


class OpMod(Enum):
    """Operator modality identifiers (shared across all projection backends)."""
    DCT = "DCT"                # Tomosynthesis 3D cone
    CBCT = "CBCT"              # Cone-beam CT 3D
    FANBEAM2D = "FANBEAM2D"    # 2D fan-beam CT
    PARALLEL2D = "PARALLEL2D"  # 2D parallel-beam CT


def opmod_from_geometry(g: Geometry) -> OpMod:
    """Infer operator modality (OpMod) from a Geometry instance."""
    if g.modality == Modality.TOMO and g.beam == BeamGeom.CONE and g.is3d:
        return OpMod.DCT
    if g.modality == Modality.CT3D and g.beam == BeamGeom.CONE and g.is3d:
        return OpMod.CBCT
    if g.modality == Modality.CT2D and g.beam == BeamGeom.FANFLAT and not g.is3d:
        return OpMod.FANBEAM2D
    if g.modality == Modality.CT2D and g.beam == BeamGeom.PARALLEL and not g.is3d:
        return OpMod.PARALLEL2D
    raise ValueError(f"Unsupported geometry: {g.modality}/{g.beam}, is3d={g.is3d}")


class GeometricOp(ABC):
    """Abstract base for all geometric operator wrappers (ASTRA, FuXSim, Raptor, etc.)."""
    def __init__(self, geometry: Geometry):
        self.geometry = geometry 

    @abstractmethod
    def project(self, vol_xyz: xp.ndarray, vx_xyz: Tuple[float, ...]) -> xp.ndarray:
        ...

    @abstractmethod
    def backproject(self, sino: xp.ndarray,
                    reco_dim_xyz: Tuple[int, ...],
                    reco_vx_xyz: Tuple[float, ...]) -> xp.ndarray:
        ...

    @abstractmethod
    def reconstruct(self, method: str, sino: xp.ndarray,
                    reco_dim_xyz: Tuple[int, ...],
                    reco_vx_xyz: Tuple[float, ...],
                    params: Optional[dict] = None) -> xp.ndarray:
        ...
