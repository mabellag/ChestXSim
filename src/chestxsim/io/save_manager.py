from typing import Optional, Dict, Any
from pathlib import Path
import hashlib, re

from chestxsim.core.data_containers import MetadataContainer, volumeData
from chestxsim.core.device import xp
from chestxsim.io.paths import (
    RESULTS_DIR,
    STEP_TO_FOLDER,
    UNITS_TO_FOLDER,
    TISSUE_TO_FOLDER,
    SPECTRUM_TO_FOLDER,
)

class SaveManager:
    """
    Manager class for saving simulation volumes and metadata in a structured and consistent way.

    - Resolves save folder structure based on step class and metadata.
    - Ensures unique save paths by hashing metadata and simulation logs.
    - Supports saving multi-channel volumes with per-tissue subfolders.
    - Writes metadata and processing logs to text files for reproducibility.

    Folder structure: results/<step>/<spectrum>/<units>/<tissue>/<id>/
    """

    def __init__(self, base_save_dir: Optional[str] = None):
        if base_save_dir is None:
            base_save_dir = RESULTS_DIR 

        
        base_path = Path(base_save_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_path


    def resolve_folder_structure(self, step_class: str, metadata: MetadataContainer) -> Dict[str, Path]:
        step_folder = STEP_TO_FOLDER.get(step_class)
        sim_id = metadata.id or "unnamed"
        components = [step_folder]

        spectrum = metadata.step_outputs.get("PhysicsEffect", {}).get("spectrum", "")
        if spectrum:
            spectrum_key = "polychromatic" if "poly" in spectrum.lower() else "monochromatic"
            components.append(SPECTRUM_TO_FOLDER[spectrum_key])

        paths = {}

        projection_done = "Projection" in metadata.step_outputs

        if not projection_done:
            units = metadata.step_outputs.get("UnitConverter", {}).get("units", "")
            units_folder = UNITS_TO_FOLDER.get(units.lower(), "")
            if units_folder:
                components.append(units_folder)

        tissue_types = metadata.step_outputs.get("TissueSegmenter", {}).get("tissue_types", [])
        if isinstance(tissue_types, str):
            tissue_types = [tissue_types]

        multichannel = len(metadata.dim) > 3 and metadata.dim[3] > 1

        if multichannel and tissue_types:
            for tissue in tissue_types:
                path_parts = components.copy()
                if tissue in TISSUE_TO_FOLDER:
                    path_parts.append(TISSUE_TO_FOLDER[tissue])
                path_parts.append(sim_id)
                full_path = self.base_dir.joinpath(*path_parts)
                paths[tissue] = self.get_unique_path(full_path, metadata)
        else:
            path_parts = components.copy()
            path_parts.append(sim_id)
            full_path = self.base_dir.joinpath(*path_parts)
            paths["default"] = self.get_unique_path(full_path, metadata)

        return paths

    def get_unique_path(self, base_path: Path, metadata: MetadataContainer) -> Path:
        """
        Ensure the save path is unique by comparing hashes of existing info.txt and log.txt files.

        If an exact match is found (same info and log hash), the existing path is reused.
        Otherwise, a new numbered path is created (e.g., <base_path>_1)
        """

        if len(metadata.dim) == 4 and metadata.dim[-1]:
            metadata.dim = metadata.dim[:3] + (1,)

        current_info_hash = self.hash_info(metadata)
        current_log_hash = self.hash_log(metadata)

        if not base_path.exists():
            return base_path

        counter = 1
        alt_path = base_path
        while True:
            info_file = alt_path / "info.txt"
            log_file = alt_path / "log.txt"
        
            if info_file.exists() and log_file.exists():
                with open(info_file, "rb") as f: 
                    existing_info_hash = hashlib.md5(f.read()).hexdigest()
                with open(log_file, "rb") as f: 
                    existing_log_hash = hashlib.md5(f.read()).hexdigest()

                if existing_info_hash == current_info_hash and existing_log_hash == current_log_hash:
                    # print(f"[SaveManager] Reusing existing folder (simulation identical): {alt_path}")
                    return alt_path

            alt_path = Path(f"{base_path}_{counter}")
            if not alt_path.exists():
                # print(f"[SaveManager] Simulation with ID '{metadata.id}' has changed — saving new results to: {alt_path}")
                return alt_path
            counter += 1

    def save_step(self, step_name: str, step_outcome: volumeData):
        folder_paths = self.resolve_folder_structure(step_name, step_outcome.metadata)
        volume = step_outcome.volume
        metadata = step_outcome.metadata
        sim_id = metadata.id or "unnamed"

        if volume.ndim == 4 and volume.shape[-1] > 1:
            tissue_types = metadata.step_outputs.get("TissueSegmenter", {}).get("tissue_types", [])
            if len(tissue_types) == volume.shape[-1]:
                for i, tissue in enumerate(tissue_types):
                    if tissue in folder_paths:
                        path = folder_paths[tissue]
                        path.mkdir(parents=True, exist_ok=True)
                        single_channel = volume[..., i]
                        self.save_volume(single_channel, path, f"{sim_id}.img")
                        self.write_metadata_to_txt(metadata, path)
                        print(f"[SaveManager] Saved {tissue} channel to: {path}")
            else:
                path = folder_paths["default"]
                path.mkdir(parents=True, exist_ok=True)
                self.save_volume(volume, path, f"{sim_id}.img")
                self.write_metadata_to_txt(metadata, path)
                print(f"[SaveManager] Saved multi-channel volume to: {path}")
        else:
            path = folder_paths["default"]
            path.mkdir(parents=True, exist_ok=True)
            self.save_volume(volume, path, f"{sim_id}.img")
            self.write_metadata_to_txt(metadata, path)
            print(f"[SaveManager] {step_name} result saved to: {path}")

    def save_volume(self, volume: Any, dir_path: str, file_name: str, dtype='<f4'):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        full_path = dir_path / file_name
        datab = xp.reshape(volume, volume.size, order='F')
        im = xp.asfortranarray(datab, dtype=dtype)
        im.tofile(str(full_path))

    def write_metadata_to_txt(self, metadata: MetadataContainer, dir_path: Path):
        info_file = dir_path / "info.txt"
        log_file = dir_path / "log.txt"

        # Explicitly formatted and consistent content
        info_content = (
            f"dim: {metadata.dim}\n"
            f"voxel_size: {metadata.voxel_size}\n"
            f"dtype: {metadata.dtype}\n"
            f"endianness: {metadata.endianness}\n"
            f"order: {metadata.order}\n"
            f"id: {metadata.id}\n"

        )
    
        with open(info_file, "w", encoding="utf-8", newline="\n") as f_info:
            f_info.write(info_content)

        log_content = "*** CHESTXSIM PROCESSING LOG ***\n\n"

        if metadata.init:
            log_content += f"init: {metadata.init}\n"        
        # then step outputs 
        for step_name, details in metadata.step_outputs.items():
            log_content += f"{step_name}: {details}\n"

        with open(log_file, "w", encoding="utf-8", newline="\n") as f_log:
            f_log.write(log_content)

    @staticmethod
    def hash_info(metadata: MetadataContainer) -> str:
        content = (
            f"dim: {metadata.dim}\n"
            f"voxel_size: {metadata.voxel_size}\n"
            f"dtype: {metadata.dtype}\n"
            f"endianness: {metadata.endianness}\n"
            f"order: {metadata.order}\n"
            f"id: {metadata.id}\n"
        )
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_text(s: str) -> str:
        # quita espacios después de comas y colapsa espacios múltiples
        s = s.replace(", ", ",")
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    @staticmethod
    def hash_log(metadata: MetadataContainer) -> str:
        content = "*** CHESTXSIM PROCESSING LOG ***\n\n"
        if metadata.init:
            content += f"init: {metadata.init}\n"
        for step_name, details in metadata.step_outputs.items():
            content += f"{step_name}: {details}\n"
        content = SaveManager._normalize_text(content)
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def save(self, vol_data: volumeData, custom_folder: str = None):
        path = self.base_dir / (custom_folder if custom_folder else vol_data.metadata.id or "unnamed")
        unique_path = self.get_unique_path(path, vol_data.metadata)
        unique_path.mkdir(parents=True, exist_ok=True)
        self.save_volume(vol_data.volume, unique_path, f"{vol_data.metadata.id or 'volume'}.img")
        self.write_metadata_to_txt(vol_data.metadata, unique_path)
        print(f"Saved final result to: {unique_path}")


