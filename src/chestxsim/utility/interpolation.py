from typing import Tuple
from chestxsim.core import xp, volumeData 
from scipy import interpolate  # this works on cpu 
import copy
import numpy as np

def interpolate_funct(
    img: np.ndarray,
    original_voxel_size: Tuple[float, float, float],
    new_voxel_size: Tuple[float, float, float],
    target_nb_pixels: Tuple[int, int, int],
):
    """
    Interpolate a 3D NumPy volume onto a new voxel grid using SciPy (CPU).

     """
    # Original volume shape and voxel size
    original_shape = img.shape
    # Create the original grid
    x = np.linspace(-1 / 2 * (original_shape[0] - 1) * original_voxel_size[0], 1 / 2 * (
        original_shape[0] - 1) * original_voxel_size[0], original_shape[0])
    y = np.linspace(-1 / 2 * (original_shape[1] - 1) * original_voxel_size[1], 1 / 2 * (
        original_shape[1] - 1) * original_voxel_size[1], original_shape[1])
    z = np.linspace(-1 / 2 * (original_shape[2] - 1) * original_voxel_size[2], 1 / 2 * (
        original_shape[2] - 1) * original_voxel_size[2], original_shape[2])

    # Create the new grid with the desired voxel size
    x_new = np.linspace(-1 / 2 * (target_nb_pixels[0] - 1) * new_voxel_size[0], 1 / 2 * (
        target_nb_pixels[0] - 1) * new_voxel_size[0], target_nb_pixels[0])
    y_new = np.linspace(y[-1] - (target_nb_pixels[1] - 1)
                        * new_voxel_size[1], y[-1], target_nb_pixels[1])
    z_new = np.linspace(-1 / 2 * (target_nb_pixels[2] - 1) * new_voxel_size[2], 1 / 2 * (
        target_nb_pixels[2] - 1) * new_voxel_size[2], target_nb_pixels[2])

    # Create the interpolation function with  interpolation
    interp_function = interpolate.RegularGridInterpolator(
        (x, y, z), img, method='linear', bounds_error=False, fill_value=0)

    # Create the meshgrid for the new volume
    x_new_mesh, y_new_mesh, z_new_mesh = np.meshgrid(
        x_new, y_new, z_new, indexing='ij')

    # Generate the new volume by interpolating on the new grid
    new_points = np.stack((x_new_mesh, y_new_mesh, z_new_mesh), axis=-1)
    resampled_volume = interp_function(new_points)

    # Output the new shape of the resampled volume
    resampled_shape = resampled_volume.shape

    print("Original shape:", original_shape)
    print("Resampled shape:", resampled_shape)
    return resampled_volume



class Interpolator():
    """
    Resample a 3D volume to a new voxel spacing and size using SciPy.

    All interpolation is done on CPU (SciPy)
    """
    def __init__(
            self, 
            target_voxel_size: Tuple[float,float,float],
            target_size: Tuple[float, float,float]):
  
        self.target_voxel_size = target_voxel_size
        self.target_size= target_size

    def __call__(self, input_volume: volumeData):
        volume = input_volume.volume
        metadata = copy.deepcopy(input_volume.metadata)

        # --- Ensure volume is 3D :: interpolator from scipy work on cpu 
        if volume.ndim == 4:
            # If last dim is a singleton (common in chestxsim volumes)
            if volume.shape[-1] == 1:
                volume = volume[..., 0]
            else:
                raise ValueError(
                    f"Interpolator expects a 3D volume, but got shape {volume.shape} "
                    "with multiple channels."
                )
        elif volume.ndim != 3:
            raise ValueError(
                f"Interpolator expects a 3D volume, but got {volume.ndim}D input."
            )
        volume_np = volume.get() if hasattr(volume, "get") else volume
        processed_volume = interpolate_funct(volume_np,  
                                             metadata.voxel_size, 
                                             self.target_voxel_size, 
                                             self.target_size)
        processed_volume = xp.asanyarray(processed_volume)
        metadata.dim = processed_volume.shape
        metadata.voxel_size = self.target_voxel_size
        return volumeData(volume=processed_volume, metadata=metadata)