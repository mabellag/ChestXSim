
"""
Core data abstractions used throughout the simulation pipeline:

    - volumeData: wraps a volumetric image and its metadata
    - Geometry and its subclasses: describe acquisition geometry
    - Physics: models X-ray source spectrum and material attenuation properties

Path handling for external resources (spectra, MAC tables) is centralized and based on
project-relative paths defined in `io_utils.path_config`.

"""

from dataclasses import dataclass, field
from typing import Dict, Union, Optional, Tuple, Any
from pathlib import Path 
import scipy.io as sio

from chestxsim.core.device import xp          
from chestxsim.io.paths import (
    MATERIALS_DIR,
    EXECUTABLES_DIR,
    MODELS_DIR,
    MAC_DIR,
    SPECTRUM_DIR,
)

@dataclass
class MetadataContainer:
    """
    Stores metadata associated with a volumeData object, including voxel spacing,
    dimensions, and step-by-step outputs recorded across the simulation pipeline
    for tacking operations.
    """
    dim: Tuple[int, ...] = (0, 0, 0)
    voxel_size: Tuple[float, ...] = (1.0, 1.0, 1.0)
    id: Optional[str] = None
    dtype: str= '<f4',
    endianness: str = "<"
    order: str = "F"
    init: Dict[str, Any] = field(default_factory=dict)
    step_outputs: Dict[str, Any] = field(default_factory=dict)
    
    def show_steps(self):
        """
        PrintS all recorded steps and their key fields if present
        """
        print("\n--- Simulation Steps ---")
        for k, v in self.step_outputs.items():
            print(f"{k}: {v}")
        
    
    def last(self, field: str, default=None):
        """
        Return the most recent occurrence of a field across all steps.
        Example:
            md.last("units") -> 'mu'
        """
        for step_name in reversed(list(self.step_outputs.keys())):
            step_data = self.step_outputs[step_name]
            if isinstance(step_data, dict) and field in step_data:
                return step_data[field]
        return default
    
  
    # def find(self, field: str, multiple: bool = False, default=None):
    #     """
    #     Search all recorded steps for a given field name.

    #     Args:
    #         field: the key to search for in each step's output
    #         multiple: if True, return a dict {step_name: value} for all matches.
    #                   if False (default), return the first match value found
    #                   in reverse step order (most recent first).
    #         default: value returned if not found.

    #     Examples:
    #         md.find("units")                -> 'mu'
    #         md.find("units", multiple=True) -> {'UnitConverter': 'mu'}
    #     """
    #     if multiple:
    #         results = {
    #             step: data[field]
    #             for step, data in self.step_outputs.items()
    #             if isinstance(data, dict) and field in data
    #         }
    #         return results if results else default

    #     # Single (latest) match, reverse order = last written
    #     for step_name in reversed(list(self.step_outputs.keys())):
    #         data = self.step_outputs[step_name]
    #         if isinstance(data, dict) and field in data:
    #             return data[field]
    #     return default

    def find(self, key: str, default: Any=None) -> Any:
        """
        Search order:
        1) Direct attribute on the container (e.g., voxel_size)
        2) 'step_outputs' dicts (latest-first), shallow keys first
        3) 'init' dict (e.g., ct_vx)
        """
        # 1) direct attribute
        if hasattr(self, key):
            value = getattr(self, key)
            if value is not None:
                return value

        # 2) init dict
        if isinstance(self.step_outputs, dict) and self.step_outputs:
            # preserve insertion order, walk from newest to oldest
            for _, data in reversed(list(self.step_outputs.items())):
                if isinstance(data, dict) and key in data:
                    return data[key]
        
        # 3) step_outputs (latest first)
        if isinstance(self.init, dict) and key in self.init:
            return self.init[key]
      

            # # 4) deep search inside step_outputs (latest-first)
            # for _, data in reversed(list(self.step_outputs.items())):
            #     found = _deep_find(data, key)
            #     if found is not _NOT_FOUND:
            #         return found

        return default
       

    

@dataclass
class volumeData:
    """
    Container for a 3D or 4D volume and its associated metadata.

    The volume can be a NumPy or CuPy array, typically shaped as (H, W, D, [T]),
    where T is the tissue/material channel dimension (optional).
    """
    volume: Any 
    metadata: MetadataContainer = field(default_factory=MetadataContainer)

    def __post_init__(self):
        if isinstance(self.metadata, dict):
            self.metadata = MetadataContainer(**self.metadata)

        
@dataclass 
class SourceSpectrum:
    """X-ray source spectrum model. 
    Handles automatic loading of X-ray spectra from .mat files.
    Supports both mono- and poly-chromatic modes.
    """

    I0: Union[float, int]
    voltage: int
    poly_flag: bool = True  # True → polychromatic, False → monochromatic
    specter_path: Optional[Path] = None
    effective_energy: Optional[int] = None  # required if poly_flag=False
    _cache: Optional[Any] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.specter_path is None:
            self.specter_path = SPECTRUM_DIR/ f"spectr{self.voltage}kVp.mat"
       
    def _load_spectrum(self) -> Any:
        """Load the spectrum from .mat file."""
        if not Path(self.specter_path).exists():
            print(f"[SourceSpectrum] File not found: {self.specter_path}")
            return xp.asarray([])
        print(f"[SourceSpectrum] Reading X-ray spectrum from {self.specter_path.resolve()}")
        try:
            spectrum_data = sio.loadmat(self.specter_path)
            spectrum = spectrum_data.get("spectrum", xp.asarray([])).flatten()
        except Exception as e:
            print(f"[SourceSpectrum] Could not load {self.specter_path}: {e}")
            return xp.asarray([])

        if spectrum.size == 0:
            print(f"[SourceSpectrum] Empty or invalid spectrum in file: {self.specter_path}")
            return xp.asarray([])

        # Truncate to the number of kVp values 
        spectrum = spectrum[1:self.voltage + 1]
        return xp.asarray(spectrum)
    
    @property
    def spectrum(self) -> Any:
        """Return the X-ray spectrum.
        - If poly_flag=True → normalized polychromatic spectrum.
        - If poly_flag=False → monochromatic spectrum at `effective_energy`.
        """
        if self._cache is not None:
            return self._cache

        spectrum = self._load_spectrum()

        if not self.poly_flag:
            # Monochromatic: use only the effective energy bin
            if self.effective_energy is None:
                raise ValueError("[SourceSpectrum] effective_energy must be set for mono mode.")
            e_eff = int(self.effective_energy)
            val_eff = spectrum[e_eff - 1] if e_eff - 1 < len(spectrum) else spectrum[-1]
            spectrum = xp.zeros_like(spectrum)
            # spectrum[e_eff - 1] = val_eff
            spectrum[e_eff - 1] = 1
        else:
            # Normalize polychromatic spectrum
            sum_spectrum = xp.sum(spectrum)
            if sum_spectrum != 1.0:
                spectrum /= sum_spectrum
        
        self._cache = spectrum
        return spectrum

    
@dataclass
class MACRepo:
    """
    Lazy loader for tissue MAC curves in cm2/g
    - Default file pattern: MAC_DIR / f"mac_{material}.txt"
    - Dot access: macs.bone   -> loads mac_bone.txt
    - Register custom data via .read(path, material) or .register(material, data)
    """
    folder: Optional[Path] = None
    _cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.folder is None:
            self.folder = Path(MAC_DIR)
        try:
            print(f"[MAC] Base folder for MACs: {self.folder.resolve()}")
        except Exception:
            print(f"[MAC] Base folder for MACs: {self.folder}")

    def register(self, material: str, data: Any) -> None:
        """Register an in-memory curve for a material (overrides disk)."""
        key = self._norm(material)
        self._cache[key] = xp.asarray(data)

    def read(self, path: Path, material: str) -> Any:
        """
        Load a curve from an arbitrary file and register it under `material`.
        Returns the loaded array.
        It can have any name format not the default  "mac_{key}.txt"
        """
        try:
            arr = xp.loadtxt(path)
        except Exception as e:
            print(f"[MAC] Could not load {path}: {e}")
            arr = xp.asarray([])
        self.register(material, arr)
        return arr


    def __getattr__(self, name: str) -> Any:
        # macs.bone  -> "bone"
        key = self._norm(name)
        return self._get_material(key)

    def _get_material(self, key: str) -> Any:
        if key in self._cache:
            return self._cache[key]
        

        path = self.folder / f"mac_{key}.txt"
        if not path.exists():
            print(f"[MAC] File not found: {path}")
            arr = xp.asarray([])
            self._cache[key] = arr
            return arr

        try:
            arr = xp.loadtxt(path)
        except Exception as e:
            print(f"[MAC] Failed to load {path}: {e}")
            arr = xp.asarray([])

        self._cache[key] = arr
        return arr

    @staticmethod
    def _norm(material: str) -> str:
        return str(material).strip().lower()
