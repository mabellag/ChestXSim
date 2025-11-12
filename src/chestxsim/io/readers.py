"""
CT Readers: DICOM and RAW

This module extends `CTReader` to handle multiple CT volume formats:

- DICOM (via pydicom):
    * Reads all slices in a folder.
    * Sorts by InstanceNumber to ensure correct z-order.
    * Returns volume in [x, y, z] (width, height, depth).
    * Extracts key metadata (dimensions, voxel size, voltage, slope/intercept).

- RAW (binary) + metadata (info.txt):
    * Reads raw bytes from a single .img file whose basename matches the folder.
    * All decoding parameters are read from info.txt (with sensible defaults):
        - dim: (H, W, D)
        - voxel_size: (sx, sy, sz)
        - dtype: e.g. "float32", "uint16", "<f4", ">i16"
        - endianness: "<" | ">" | "=" (optional; ignored if dtype already encodes it)
        - order: "F" | "C"             (default "F")
        - header_bytes: int            (default 0)
        - scale: float                 (default 1.0)
        - intercept: float             (default 0.0)
        - id: str

Both readers return:
    volumeData(volume, metadata)

Where `volume` is a NumPy/CuPy array (xp backend) in [x, y, z].
"""

import os
import re
import copy
import pydicom
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Optional, List
from chestxsim.core.device import xp
from chestxsim.core.data_containers import volumeData, MetadataContainer
from chestxsim.io.paths import *


class CTReader(ABC):
    """
    Abstract base class for CT data readers
    
    Returns:
        volumeData object containing:
        - volume: CuPy array (if GPU detected) or NumPy array (CPU fallback) float-32
        - metadata: MetadataContainer with scan parameters
    
    Output volume coordinate system: [x,y,z] where:
            z (top to bottom)
            |
            | __ x (side to side)
            /
        y (anterior-posterior)
    
    """

    def __init__(
        self, 
        convert_to_HU: bool = False,
        clip_values: Optional[Tuple[float, float]] = None      #e.g. [-1024.0, 3000.0]
    ):
        self.convert_to_HU = convert_to_HU
        self.clip_values = clip_values
        
      
    
    def read(self, folder:str) -> volumeData:
        """
        Read a case folder and optionally:
          - convert to HU using metadata slope/intercept
          - clip intensities to a given range
        """
     
        data = self.load_case(folder)

        # Stamp once; never overwrite if already present
        md = data.metadata
        md.init.setdefault("ct_dim", tuple(md.dim))
        md.init.setdefault("ct_vx", tuple(md.voxel_size))

            
        # Apply slope and intercept to convert to HU
        # HU conversion uses safe defaults from init
        if self.convert_to_HU:
            slope = float(md.init.get("slope", 1.0))
            intercept = float(md.init.get("intercept", 0.0))
            data.volume = data.volume * slope + intercept
            # data.metadata.step_outputs["original_input_units"] = "HU"
        
        # Apply clipping 
        if self.clip_values is not None:
            min_val, max_val = self.clip_values
            data.volume = xp.clip(data.volume, min_val, max_val)
            data.metadata.step_outputs["clip_values"] = f"[{min_val},{max_val}]"
        
        return data


    @abstractmethod
    def load_case(self, folder:str) -> volumeData:
        """Load a case folder and return volumeData."""
        pass

    @staticmethod
    @abstractmethod
    def read_metadata():
        """Read metadata for the case."""
        pass

    @staticmethod
    @abstractmethod
    def read_volume():
        """Read the actual volume array."""
        pass

class DicomReader(CTReader):
    """
    DICOM CT data reader using pydicom.

    Notes:
        Assemble the DICOM stack and return [x, y, z].
        We sort slices by InstanceNumber; if missing, fallback to filename order.
    """

    def load_case(self, dicomFolder) -> volumeData:
        metadata = self.read_metadata(dicomFolder)
        volume = self.read_volume(dicomFolder,
                                metadata.dim,
                                metadata.dtype
                                )
        return volumeData(volume, metadata)
    
    @staticmethod
    def read_metadata(dicomFolder:str)-> MetadataContainer:
        """Extract basic metadata from the first DICOM file."""
        files = sorted([f for f in os.listdir(dicomFolder)])
    
        # print(os.path.join(dicomFolder,files[0]))
        first_dicom = pydicom.dcmread(os.path.join(dicomFolder,files[0]))
        
        # get dimensions (H,W,D)
        metadata = MetadataContainer()
        # metadata.Input["dim"]= (first_dicom.Rows, first_dicom.Columns, len(files))
        metadata.dim = (first_dicom.Rows, first_dicom.Columns, len(files))
        
        # get voxel size
        pixspa = first_dicom.PixelSpacing
        if hasattr(first_dicom, "SpacingBetweenSlices"):
            slice_spacing = abs(first_dicom.SpacingBetweenSlices)
        else:
            slice_spacing = abs(first_dicom.SliceThickness / 2)

       
        metadata.voxel_size = tuple(round(x, 3) for x in (pixspa[0], pixspa[1], slice_spacing))
        # metadata.init["dim"] = metadata.dim
        # metadata.init["voxe_size"] = metadata.voxel_size
        metadata.init["voltage"] = float(first_dicom.KVP)
        metadata.init["slope"] = float(first_dicom.RescaleSlope)
        metadata.init["intercept"] = float(first_dicom.RescaleIntercept)
        metadata.dtype = str(first_dicom.pixel_array.dtype)
       
        # store id 
        parts = dicomFolder.split(os.sep)
        if "inputs" in parts:
            index = parts.index("inputs")
            target_folder = "_".join(parts[index + 1:])
        else:
            target_folder = None
        
        metadata.id = target_folder
        return metadata 
    
    # @staticmethod
    # def read_volume(dicomFolder:str, shape: tuple, dtype: str):
    #     """
    #     Assemble the DICOM stack and return [x, y, z].
    #     We sort slices by InstanceNumber; if missing, fallback to filename order.
    #     """
    #     volume = xp.zeros(shape=  shape, dtype = dtype)
    #     files = sorted([f for f in os.listdir(dicomFolder)])
    #     # check here instance number 
    #     for ct_slice in files:
    #         dicom = pydicom.dcmread(os.path.join(dicomFolder, ct_slice))
    #         volume[:, :, dicom.InstanceNumber - 1] = xp.asanyarray(dicom.pixel_array)     
        
    #     return xp.swapaxes(volume, 0,1)

    @staticmethod
    def read_volume(dicom_folder: str, shape: tuple, dtype: str):
        """
        Assemble the DICOM stack and return [x, y, z].
        Sorts slices by InstanceNumber if available; otherwise uses filename order.
        """
        files = sorted([f for f in os.listdir(dicom_folder) if not f.startswith(".")])
        first = pydicom.dcmread(os.path.join(dicom_folder, files[0]))

        # Check if InstanceNumber exists
        has_instance = hasattr(first, "InstanceNumber")

        if has_instance:
            # Sort by InstanceNumber
            dicoms = [pydicom.dcmread(os.path.join(dicom_folder, f)) for f in files]
            dicoms.sort(key=lambda d: int(d.InstanceNumber))
        else:
            # Just use sorted filenames
            dicoms = [pydicom.dcmread(os.path.join(dicom_folder, f)) for f in files]

        # Stack slices
        volume = xp.zeros(shape=shape, dtype=dtype)
        for i, ds in enumerate(dicoms):
            volume[:, :, i] = xp.asanyarray(ds.pixel_array)

        return xp.swapaxes(volume, 0, 1)

class RawReader(CTReader):
    """
    RAW binary CT data reader.

    Single-tissue (per case):
        - Expects ONE .img file and ONE info.txt inside a case folder.
        - `load_case(<case_folder>)` → returns `volumeData` with:
            volume: xp.ndarray of shape (H, W, D) in [y, x, z] order internally,
                    returned to the pipeline as [x, y, z] if you swap later.
            metadata: MetadataContainer parsed from info.txt / log.txt.
        - Folder layout:
            <case_folder>/
                info.txt        # required (see keys below)
                log.txt         # optional
                <case_name>.img # raw binary payload

    Multi-tissue (composite case):
        - Expects multiple case folders (one per tissue), each with its own .img + info.txt.
        - Topology: sibling folders named by tissue (per TISSUE_TO_FOLDER mapping), each
        containing a subfolder named <case_id>.
        - `load_multi_tissue(base_path, case_id, combine_method)`:
            combine_method = "sum":
                returns volume of shape (H, W, D)  # channel-wise sum across tissues
            combine_method = "stack":
                returns volume of shape (H, W, D, T)  # last dim T = number of tissues
        Metadata:
            - `metadata.tissue_types` lists the tissue order used.
            - `metadata.dim` is set to the combined volume shape.
        - Folder layout (example):
            <base_path_parent>/
                tissue1 /
                    <case_id>/
                        info.txt
                        log.txt   (optional)
                        <case_id>.img
                tissue2/
                    <case_id>/
                        info.txt
                        <case_id>.img
               
    Required info.txt keys:
        dim: (H, W, D)
        voxel_size: (sx, sy, sz)

    Optional info.txt keys (defaults):
        dtype: "float32" | "uint16" | "<f4" | ">i16" | ...   (default "<f4")
        endianness: "<" | ">" | "="                          (ignored if dtype encodes it)
        order: "F" | "C"                                     (default "F")
        scale: float                                         (default 1.0)
        intercept: float                                     (default 0.0)
        id: str

    Notes:
        - dtype may embed endianness (e.g., "<f4"); if not, `endianness` is applied.
        - RAW volumes are reshaped using `order`; default "F" matches common scientific exports.
        - If `convert_to_HU=True` in the base reader, the final volume is scaled by
        `slope` and `intercept` from metadata if present.
    """


    def load_case(self, folder) -> volumeData:
        _id = os.path.basename(folder)

        # Remove trailing _<number> if present (before .img)
        fileterd_id = re.sub(r'_\d+$', '', _id)
      

        metadata_info_path = os.path.join(folder, "info.txt")
        metadata_log_path = os.path.join(folder, "log.txt")
        metadata = self.read_metadata(metadata_info_path, metadata_log_path)


        volume_path = os.path.join(folder, f"{fileterd_id}.img")
        volume = self.read_volume(volume_path, metadata.dim)

        return volumeData(volume, metadata)

    @staticmethod  
    def read_metadata(info_path: Union[str,Path], log_path:Union[str,Path]) -> MetadataContainer:
        info_path, log_path = Path(info_path), Path(log_path)
        md = MetadataContainer()

        if not info_path.exists():
            raise FileNotFoundError(f"[RawReader] Missing info.txt at {info_path}")
        
        with info_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or ":" not in line:
                    continue

                key, value = (s.strip() for s in line.split(":", 1))
                try:
                    value = eval(value)
                except Exception:
                    pass
                if key == "dim":
                    md.dim = tuple(value)
                elif key == "voxel_size":
                    md.voxel_size = tuple(value)
                elif key == "id":
                    md.id = str(value)
                elif key == "dtype":
                    md.dtype = str(value)
                elif key == "endianness":
                    md.endianness = str(value)
                elif key == "order":
                    md.order = str(value).upper()
                # elif key == "scale":
                #     md.init["scale"] = float(value)
                # elif key == "intercept":
                #     md.init["intercept"] = float(value)
                else:
                    # keep any extra fields for traceability
                    md.init[key] = value

        # validate required fields
        if not getattr(md, "dim", None):
            raise ValueError(f"[RawReader] 'dim' is required in {info_path}")
        if not getattr(md, "voxel_size", None):
            raise ValueError(f"[RawReader] 'voxel_size' is required in {info_path}")

        # # apply defaults for optionals if missing
        # md.setdefault("dtype", "<f4")
        # md.setdefault("endianness", None) 
        # md.setdefault("order", "F")
        # md.setdefault("scale", 1.0)
        # md.setdefault("intercept", 0.0)
        if log_path.exists():
            with log_path.open("r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    
                    if not line or line.startswith("***") or ":" not in line:
                        continue

                    step_name, details = (s.strip() for s in line.split(":", 1))
                    try:
                        parsed = eval(details)
                    except Exception:
                        parsed = details  # leave as string if not a literal

                    if step_name.lower() == "init":
                        if isinstance(parsed, dict):
                            md.init.update(parsed)
                        else:
                            md.init["raw_init"] = parsed
                    else:
                        md.step_outputs[step_name] = parsed

                return md

    
    @staticmethod
    def read_volume(
        file_path: str | Path,
        shape: tuple,
        dtype: str = "<f4",
        endianness: str | None = None,
        order: str = "F",
       
        ):
        file_path = Path(file_path)

        # Normalize dtype and apply endianness if needed
        s = dtype.strip().lower()
        if not s.startswith(("<", ">", "=", "|")):
            aliases = {
                "float32": "f4", "float": "f4", "single": "f4",
                "float64": "f8", "double": "f8",
                "int16": "i2",  "uint16": "u2",
                "int32": "i4",  "uint32": "u4",
                "int8": "i1",   "uint8": "u1",
            }
            base = aliases.get(s, s)
            prefix = endianness if endianness in ("<", ">", "=", "|") else "<"
            s = prefix + base
        np_dtype = xp.dtype(s)
        with open(file_path, "rb") as fid:
            data = xp.fromfile(fid, np_dtype)
        return data.reshape(shape, order=order)

        
    def load_multi_tissue(self, base_path: Union[str, Path], 
                     case_id: str,
                     combine_method: str = "sum") -> volumeData:
        """
        Load and combine multiple tissue types by case ID.
        
        Args:
            base_path: Path like "results/CT_converted/density" or "results/CT_converted" 
            case_id: Case identifier like "NODULO_S18_S20"
            tissues: List of tissues to load. If None, auto-detect from folders
            combine_method: "sum" or "stack"
            
        Returns:
            Combined volumeData with multi-tissue volume
        """
        base_path = Path(base_path)
        
        # Find tissue folders with the case ID
        tissue_folders = self.find_tissue_folders_by_id(base_path, case_id)
       
        if not tissue_folders:
            raise ValueError(f"No tissue folders found for case {case_id} in {base_path}")
        
        # Load all tissues
        loaded_tissues = []
        tissue_names = []
        reference_metadata = None
        
        for tissue_name, folder_path in tissue_folders.items():
            tissue_data = self.load_case(folder_path)
            loaded_tissues.append(tissue_data.volume)
            tissue_names.append(tissue_name)
            
            if reference_metadata is None:
                reference_metadata = copy.deepcopy(tissue_data.metadata)
        
        # Combine volumes
        if combine_method == "sum":
            combined_volume = sum(loaded_tissues)
        elif combine_method == "stack":
            loaded_tissues = [xp.squeeze(vol) for vol in loaded_tissues] 
            combined_volume = xp.stack(loaded_tissues, axis=-1) 
        else:
            raise ValueError(f"Unknown combine_method: {combine_method}")
        
        # Update metadata
        combined_metadata = copy.deepcopy(reference_metadata)
        combined_metadata.tissue_types = tissue_names
        combined_metadata.dim = combined_volume.shape
        return volumeData(combined_volume, combined_metadata)
    
    @staticmethod
    def find_tissue_folders_by_id(base_path: Path, case_id: str) -> Dict[str, Path]:
        """Find tissue folders containing the case ID"""
        tissue_folders = {}

        # look for any tissue folder containing case_id
        search_dirs = [base_path, base_path.parent]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            for item in search_dir.iterdir():
                if item.is_dir() and item.name in TISSUE_TO_FOLDER.values():
                    case_path = item / case_id
                    if case_path.exists():
                        # Find tissue key from folder name
                        tissue_key = next((k for k, v in TISSUE_TO_FOLDER.items() if v == item.name), item.name)
                        tissue_folders[tissue_key] = case_path
                        try:
                            print(f"[RawReader] Found tissue '{tissue_key}' at {case_path.resolve()}")
                        except Exception:
                            print(f"[RawReader] Found tissue '{tissue_key}' at {case_path}")
        
        return tissue_folders

    








# class RawReader(CTReader):
#     """
#     RAW binary CT data reader (single .img + info.txt per case).

#     Expected folder layout:
#         <case_folder>/
#             info.txt      # metadata driving the read (see keys below)
#             log.txt       # optional pipeline log
#             <case_name>.img

#     Required `info.txt` keys:
#         dim: (H, W, D)
#         voxel_size: (sx, sy, sz)

#     Optional `info.txt` keys (with defaults):
#         dtype: "float32" | "uint16" | "<f4" | ">i16" | ...   (default "<f4")
#         endianness: "<" | ">" | "="                          (ignored if dtype encodes it)
#         order: "F" | "C"                                     (default "F")
#         scale: float                                         (default 1.0)
#         intercept: float                                     (default 0.0)
#         id: str

#     Notes:
#         it allow loading multi-tissue volumes using the method `load_`
#         it expects the folders as 
#     """

#     def load_case(self, folder) -> volumeData:
#         _id = os.path.basename(folder)

#         # Remove trailing _<number> if present (before .img)
#         fileterd_id = re.sub(r'_\d+$', '', _id)
      

#         metadata_info_path = os.path.join(folder, "info.txt")
#         metadata_log_path = os.path.join(folder, "log.txt")
#         metadata = self.read_metadata(metadata_info_path, metadata_log_path)


#         volume_path = os.path.join(folder, f"{fileterd_id}.img")
#         volume = self.read_volume(volume_path, metadata.dim)

#         return volumeData(volume, metadata)

#     @staticmethod  
#     def read_metadata(info_path: str, log_path) -> MetadataContainer:
#         metadata = MetadataContainer()
#         with open(info_path, "r") as f:
#             for line in f:
#                 line = line.strip()
#                 if not line or ":" not in line:
#                     continue

#                 key, value = line.split(":", 1)
#                 key = key.strip()
#                 value = value.strip()

#                 try:
#                     value = eval(value)
#                 except Exception:
#                     pass

#                 # Assign only known top-level fields
#                 if key == "dim":
#                     metadata.dim = value
#                 elif key == "voxel_size":
#                     metadata.voxel_size = value
#                 # elif key == "units":
#                 #     metadata.units = value
#                 elif key == "id":
#                     metadata.id = value
#                 # elif key == "tissue_types":
#                 #     metadata.tissue_types = value
#                 else:
#                      metadata.init[key] = value  # anything unknown goes into 'init'
        
#         # Read log.txt
#         if os.path.exists(log_path):
#             with open(log_path, "r", encoding="utf-8") as f:
#                 for line in f:
#                     if ":" in line and not line.startswith("***"):
#                         step_name, details = line.strip().split(":", 1)
#                         step_name = step_name.strip()
#                         try:
#                             step_details = eval(details.strip())
#                         except Exception:
#                             step_details = details.strip()
#                         metadata.step_outputs[step_name] = step_details

            

#         return metadata

    
#     @staticmethod
#     def read_volume(file_path:str, shape: tuple, dtype: str = '<f4'):
#         with open(file_path, 'rb') as fid:
#             data = xp.fromfile(fid, dtype)       
#         return data.reshape(shape, order='F')
    
    
#     def load_multi_tissue(self, base_path: Union[str, Path], 
#                      case_id: str,
#                      combine_method: str = "sum") -> volumeData:
#         """
#         Load and combine multiple tissue types by case ID.
        
#         Args:
#             base_path: Path like "results/CT_converted/density" or "results/CT_converted" 
#             case_id: Case identifier like "NODULO_S18_S20"
#             tissues: List of tissues to load. If None, auto-detect from folders
#             combine_method: "sum" or "stack"
            
#         Returns:
#             Combined volumeData with multi-tissue volume
#         """
#         base_path = Path(base_path)
        
#         # Find tissue folders with the case ID
#         tissue_folders = self.find_tissue_folders_by_id(base_path, case_id)
       
#         if not tissue_folders:
#             raise ValueError(f"No tissue folders found for case {case_id} in {base_path}")
        
#         # Load all tissues
#         loaded_tissues = []
#         tissue_names = []
#         reference_metadata = None
        
#         for tissue_name, folder_path in tissue_folders.items():
#             tissue_data = self.load_case(folder_path)
#             loaded_tissues.append(tissue_data.volume)
#             tissue_names.append(tissue_name)
            
#             if reference_metadata is None:
#                 reference_metadata = copy.deepcopy(tissue_data.metadata)
        
#         # Combine volumes
#         if combine_method == "sum":
#             combined_volume = sum(loaded_tissues)
#         elif combine_method == "stack":
#             loaded_tissues = [xp.squeeze(vol) for vol in loaded_tissues] 
#             combined_volume = xp.stack(loaded_tissues, axis=-1) 
#         else:
#             raise ValueError(f"Unknown combine_method: {combine_method}")
        
#         # Update metadata
#         combined_metadata = copy.deepcopy(reference_metadata)
#         combined_metadata.tissue_types = tissue_names
#         combined_metadata.dim = combined_volume.shape
#         return volumeData(combined_volume, combined_metadata)
    
#     @staticmethod
#     def find_tissue_folders_by_id(base_path: Path, case_id: str) -> Dict[str, Path]:
#         """Find tissue folders containing the case ID"""
#         tissue_folders = {}

#         # look for any tissue folder containing case_id
#         search_dirs = [base_path, base_path.parent]
        
#         for search_dir in search_dirs:
#             if not search_dir.exists():
#                 continue
                
#             for item in search_dir.iterdir():
#                 if item.is_dir() and item.name in TISSUE_TO_FOLDER.values():
#                     case_path = item / case_id
#                     if case_path.exists():
#                         # Find tissue key from folder name
#                         tissue_key = next((k for k, v in TISSUE_TO_FOLDER.items() if v == item.name), item.name)
#                         tissue_folders[tissue_key] = case_path
#                         try:
#                             print(f"[RawReader] Found tissue '{tissue_key}' at {case_path.resolve()}")
#                         except Exception:
#                             print(f"[RawReader] Found tissue '{tissue_key}' at {case_path}")
        
#         return tissue_folders

    



