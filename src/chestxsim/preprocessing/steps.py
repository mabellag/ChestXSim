import os, copy 
from typing import Optional, Union, List, Any
from pathlib import Path
from chestxsim.core.data_containers import volumeData
from chestxsim.core.geometries import Geometry  
from chestxsim.core.device import xp
from chestxsim.io.paths import MAC_DIR, MODELS_DIR, RESULTS_DIR
from chestxsim.io.save_manager import SaveManager
from  chestxsim.utility.ops_utils import ensure_4d
from chestxsim.utility.energy import compute_effective_energy
from . import functional  as F

__all__ = [
    "BedRemover",
    "AirCropper",
    "VolumeExtender",
    "VolumeFlipper",
    "VolumeRotate",
    "TissueSegmenter",
    "UnitConverter",
]

class BedRemover:
    def __init__(self, threshold: Optional[str] = None, save_mask: bool = True, model_name: Optional[str] = None):
        self.threshold = threshold
        self.save_mask = save_mask
        # self.model_path = model_path or os.path.join("..", "materials", "models", "model_BedSeg_full_model.pt")
        self.model_path = MODELS_DIR / model_name  if model_name else MODELS_DIR / "model_BedSeg_full_model.pt"

    def __call__(self, ct_data: volumeData) -> volumeData:
        volume = ct_data.volume if ct_data.volume.ndim == 4 else ensure_4d(ct_data.volume)
        metadata = copy.deepcopy(ct_data.metadata)

        if self.threshold is not None:
            mask = F.get_stretcher_mask_analytical(volume[..., 0], self.threshold)
        else:
            mask = F.get_stretcher_mask_dl(volume[..., 0], 1, self.model_path)

        if self.save_mask:
            saver = SaveManager()
            filename = (
                f"{metadata.id}_vx_{metadata.voxel_size[0]}_{metadata.voxel_size[1]}_{metadata.voxel_size[2]}"
                f"_dim_{mask.shape[0]}_{mask.shape[1]}_{mask.shape[2]}.img"
            )
            
            save_dir = Path(RESULTS_DIR) / "CT_bed_mask" 
            print(f"[BedRemover] Saving mask to: {save_dir/filename}")
            save_dir.mkdir(parents=True, exist_ok=True)
            saver.save_volume(mask, save_dir, filename)

        processed_volume = ensure_4d(F.remove_bed(volume[..., 0], mask))
        metadata.step_outputs[self.__class__.__name__] = {
            "bed_removed": True,
            "method": "analytical" if self.threshold is not None else "dl"
        }
        
        return volumeData(volume=processed_volume, metadata=metadata)

class AirCropper:
    def __init__(self, axis: int, tol: int = 5, delta: int = 5, channel: Optional[int] = 0):
        self.axis = axis
        self.tol = tol
        self.delta = delta
        self.c = channel

    def __call__(self, ct_data: volumeData) -> volumeData:
        volume = ensure_4d(ct_data.volume)
        metadata = copy.deepcopy(ct_data.metadata)

        _, indx = F.crop_air(volume[..., self.c], self.axis, self.tol, self.delta)

        if self.axis == 0:
            processed_volume = volume[indx[0]:indx[1], :, :, :]
        elif self.axis == 1:
            processed_volume = volume[:, indx[0]:indx[1], :, :]
        elif self.axis == 2:
            processed_volume = volume[:, :, indx[0]:indx[1], :]
        else:
            raise ValueError(f"Invalid axis {self.axis} for 4D volume. Must be 0, 1, or 2.")

      
        metadata.dim = processed_volume.shape
        metadata.step_outputs[self.__class__.__name__] = {
            "axis": self.axis,
            "crop_indices": indx
        }
        return volumeData(volume=processed_volume, metadata=metadata)

class VolumeExtender:
    def __init__(self,
                 ext_vals_mm: Optional[list[Union[int, float]]] = None,
                 target_height: Optional[Union[int, float]] = None,
                 geometry: Optional[Union[Geometry]] = None,
                 chest_center: Optional[int] = 150,
                 save=False):
        self.ext_vals_mm = ext_vals_mm
        self.target_height = target_height
        self.geometry = geometry
        self.chest_center = chest_center
        self.save = save

    def __call__(self, ct_data: volumeData) -> volumeData:
        volume = ensure_4d(ct_data.volume)
        metadata = copy.deepcopy(ct_data.metadata)

        if self.ext_vals_mm is not None:
            ext_vals_mm = self.ext_vals_mm
        elif self.target_height is not None:
            ext_vals_mm = F.compute_extension_vals_from_target_height(
                volume[..., 0], metadata.voxel_size, self.target_height, self.chest_center)
        elif self.geometry is not None:
            ext_vals_mm = F.compute_extension_vals_from_geometry(
                volume[..., 0], metadata.voxel_size, self.geometry, self.chest_center)
        else:
            raise ValueError("Must provide `ext_vals_mm`, `target_height`, or `geometry`.")

        processed_volume = xp.stack([
            F.extend_volume(volume[..., i], ext_vals_mm, metadata.voxel_size)
            for i in range(volume.shape[-1])
        ], axis=-1)

        # metadata.Preprocessing.update({
        #     "dim": processed_volume.shape,
        #     "extension_mm": ext_vals_mm
        # })
        metadata.dim = processed_volume.shape
        metadata.step_outputs[self.__class__.__name__] = {
            "extension_mm": ext_vals_mm
        }

        return volumeData(volume=processed_volume, metadata=metadata)

class VolumeFlipper:
    def __init__(self, axis: int = 2):
        self.axis = axis

    def __call__(self, ct_data: volumeData) -> volumeData:
        volume = ensure_4d(ct_data.volume)
        metadata = copy.deepcopy(ct_data.metadata)
        processed_volume = F.flip(volume, self.axis)
        metadata.step_outputs[self.__class__.__name__] = {
            "chest_to_detector": True,
            "flip_axis": self.axis
        }
        return volumeData(volume=processed_volume, metadata=metadata)


class VolumeRotate:
    def __init__(self, 
                 angle: Optional[float] = None,
                 angle_from_range: Optional[List[float]] = None,
                 axis: int = 2):
        """
        Rotate the input CT volume around the specified axis.

        Args:
            angle (float, optional): Fixed rotation angle in degrees.
            angle_from_range (list, optional): [min, max] range to sample a random angle (degrees).
            axis (int): Axis around which to rotate:
                        0 → (y, z), 1 → (x, z), 2 → (x, y)
        """ 
        self.angle = angle
        self.angle_from_range = angle_from_range
        self.axis = axis

    def __call__(self, ct_data: volumeData) -> volumeData:
        volume = ensure_4d(ct_data.volume)
        metadata = copy.deepcopy(ct_data.metadata)

        # Determine angle
        if self.angle is not None:
            angle = self.angle
        elif self.angle_from_range is not None:
            angle = xp.round(xp.random.uniform(*self.angle_from_range), 2)
        else:
            raise ValueError("Either `angle` or `angle_from_range` must be provided.")

        # Define rotation axes
        axis_map = {
            0: (1, 2),  # rotate in (y, z)
            1: (0, 2),  # rotate in (x, z)
            2: (0, 1)   # rotate in (x, y)
        }
        axes = axis_map.get(self.axis)
        if axes is None:
            raise ValueError(f"Invalid axis {self.axis}. Must be 0, 1, or 2.")

        processed_volume = F.rotate(volume, angle=angle, axes=axes)

        metadata.dim = processed_volume.shape
        metadata.step_outputs[self.__class__.__name__] = {
            "rotation_applied": True,
            "rotation_angle_degrees": angle,
            "rotation_axes": axes
        }

        return volumeData(volume=processed_volume, metadata=metadata)


class TissueSegmenter:
    def __init__(self,
                 threshold: Optional[int] = None,
                 model_path: Optional[str] = None,
                 tissue_types: List[str] = ["bone", "soft"],
                 tissue_masks: Optional[Any] = None,
                 save_mask: Optional[bool] = True):
        
        self.bone_threshold = threshold
        self.model_path = model_path or MODELS_DIR/ "model_fine_tune_vf_2.pt"
        self.tissue_types = tissue_types
        self.save_masks = save_mask
        self.tissue_masks = tissue_masks

    def __call__(self, ct_data: volumeData) -> volumeData:
        volume = ct_data.volume if ct_data.volume.ndim == 3 else ct_data.volume[..., 0]
        metadata = copy.deepcopy(ct_data.metadata)

        method = "analytical" if self.bone_threshold is not None else "dl"

        if self.tissue_masks is not None:
            tissue_masks = self.tissue_masks
        else:
            if self.bone_threshold is not None:
                binary_mask_bone = F.get_bone_mask_analytical(volume, self.bone_threshold)
            else:
                binary_mask_bone = F.get_bone_mask_dl(volume, 1, self.model_path)

            binary_soft_mask = xp.where((binary_mask_bone == 0), 1, 0)
            tissue_masks = xp.stack([binary_mask_bone, binary_soft_mask], axis=-1) # => (H, W, D, T)

        if self.save_masks:
            print("save mask tissue")
            saver = SaveManager()
            for i, tissue_type in enumerate(self.tissue_types):
                mask = tissue_masks[..., i]

                filename = (
                    f"{metadata.id}_{tissue_type}_{method}_vx_"
                    f"{metadata.voxel_size[0]}_{metadata.voxel_size[1]}_{metadata.voxel_size[2]}"
                    f"_dim_{mask.shape[0]}_{mask.shape[1]}_{mask.shape[2]}.img"
                )

                save_dir = Path(RESULTS_DIR) / "CT_tissue_masks" / method
                save_dir.mkdir(parents=True, exist_ok=True)
                full_path = save_dir / filename
                print(f"[TissueSegmenter] Saving {tissue_type} mask ({method}) to: {full_path}")
                saver.save_volume(mask, save_dir, filename)


        processed_volume = F.segment_volume(volume, tissue_masks)
        
        # metadata.Preprocessing.update({
        #     "dim": processed_volume.shape,
        #     "tissue segmented": ["True", "0" if self.bone_threshold is not None else "1"],
        #     "tissue_type": self.tissue_types
        # })

        metadata.dim = processed_volume.shape
        # metadata.tissue_types = self.tissue_types
        metadata.step_outputs[self.__class__.__name__] = {
            "tissue_segmented": True,
            "method": "analytical" if self.bone_threshold is not None else "dl",
            "tissue_types": self.tissue_types
        }

        return volumeData(volume=processed_volume, metadata=metadata)

class UnitConverter:
    def __init__(self,
                 units: str,
                 tissue_types: Optional[List[str]] = None,
                 mu_factor: Union[float, List[float]] = 1.0,
                 e_eff: Optional[int] = None,
                 voltage: Optional[int] = None):
        self.units = units.lower()
        self.tissue_types = tissue_types or []
        self.e_eff = e_eff
        self.voltage = voltage
        self.mu_factor = mu_factor
        self._mac_path = MAC_DIR

    def __call__(self, ct_data: volumeData) -> volumeData:
        volume = ensure_4d(ct_data.volume)
        metadata = copy.deepcopy(ct_data.metadata)

        if self.e_eff is not None:
            e_eff = self.e_eff
        elif self.voltage is not None:
            e_eff = xp.round(compute_effective_energy(self.voltage), 2)
        else:
            e_eff = xp.round(compute_effective_energy(metadata.init["voltage"]), 2)  

        mac_water = xp.loadtxt(self._mac_path / 'mac_water.txt')
        mac_eff_water = mac_water[int(e_eff - 1)]

        if self.units == "mu":
            if volume.shape[-1] == 1:
                factor = self.mu_factor if isinstance(self.mu_factor, (int, float)) else self.mu_factor[0]
                processed_volume = ensure_4d(factor * F.convert_HU_to_mu(volume[..., 0], mac_eff_water))
            else:
                if isinstance(self.mu_factor, list):
                    if len(self.mu_factor) != volume.shape[-1]:
                        raise ValueError("Mismatch in mu_factor list length.")
                    processed_volume = xp.stack([
                        self.mu_factor[i] * F.convert_HU_to_mu(volume[..., i], mac_eff_water)
                        for i in range(volume.shape[-1])
                    ], axis=-1)
                else:
                    processed_volume = xp.stack([
                        self.mu_factor * F.convert_HU_to_mu(volume[..., i], mac_eff_water)
                        for i in range(volume.shape[-1])
                    ], axis=-1)

            processed_volume = xp.sum(processed_volume, axis=-1, keepdims=True)
            # print(processed_volume.shape)


        elif self.units == "density":
            if not self.tissue_types or len(self.tissue_types) != volume.shape[-1]:
                raise ValueError("Mismatch between tissue_types and volume channels.")

            macs = [
                xp.loadtxt(self._mac_path / f'mac_{tissue}.txt')
                for tissue in self.tissue_types
            ]

            processed_volume = xp.stack([
                F.convert_HU_to_density(
                    volume[..., i],
                    macs[i][int(e_eff - 1)],
                    mac_eff_water,
                    self.mu_factor[i] if isinstance(self.mu_factor, list) else self.mu_factor
                ) for i in range(volume.shape[-1])
            ], axis=-1)
        else:
            raise ValueError(f"Unsupported units: '{self.units}'. Choose 'mu' or 'density'.")

        metadata.dim = processed_volume.shape
        metadata.step_outputs[self.__class__.__name__] ={
            "units":self.units,
            "e_eff": e_eff.item() if hasattr(e_eff, "item") else e_eff,
            "mac_eff_water": mac_eff_water.item() if hasattr(mac_eff_water, "item") else mac_eff_water,
            "tissue_type": self.tissue_types,
            "mu_factor": self.mu_factor

        }

        return volumeData(volume=processed_volume, metadata=metadata)

# class change_vx_size():
#     def __init__()