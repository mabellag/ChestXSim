from chestxsim.core.data_containers import *
from chestxsim.core.device import xp
from chestxsim.core.geometries import Geometry, TomoGeometry
from typing import Optional 

def ensure_4d(volume: Any) -> Any:
    """ Converts 3D volume (H,W,D) to 4D volume (H,W,D,T) where T=1 """
    if volume.ndim == 3:
        volume = volume[..., xp.newaxis]  # add tissue dimension at the end
    return volume

def apply_channelwise(fn):
    """
    Decorator that applies a function channel-wise on a 4D volume.

    It assumes that the first argument to the function is a 3D or 4D volume.
    If the input is 3D (H, W, D), it is first expanded to 4D (H, W, D, 1).
    The function is then applied separately to each 3D channel (volume[..., i]),
    and the results are stacked along the last axis to return a 4D result.

    This is useful for functions that are defined for single-channel volumes,
    but need to be applied to multi-channel (e.g., multi-tissue) data.

    Args:
        fn (Callable): A function that accepts a 3D volume as its first argument,
                       followed by any number of positional and keyword arguments.

    Returns:
        Callable: A wrapped version of the input function that supports 4D input.
    """
    def channelwise_wrapper(volume, *args, **kwargs):
        volume = ensure_4d(volume)
        return xp.stack(
            [fn(volume[..., i], *args, **kwargs) for i in range(volume.shape[-1])],
            axis=-1
        )
    return channelwise_wrapper


# def clean_gpu(fn):
#     def gpu_wrapper(*args, **kwargs):
#         xp._default_memory_pool.free_all_blocks()
#         xp.get_default_pinned_memory_pool().free_all_blocks()

#         result = fn(*args, **kwargs)

#         xp._default_memory_pool.free_all_blocks()
#         xp.get_default_pinned_memory_pool().free_all_blocks()

#         return result
#     return gpu_wrapper


# @clean_gpu
@apply_channelwise
def project(volume, opt, vol_vx):
    return opt.project(volume, vol_vx)

def build_mac_matrix(E_ENERGY: int, tissue_types: list[str], macs: MACRepo) -> Any:
    """
    Build a MAC (Mass Attenuation Coefficient) matrix for the specified tissues.

    Args:
        E_ENERGY (int): Number of energy points to include (e.g., len(spectrum)).
        macs (MACRepo): Repository providing access to MAC curves.
        tissue_types (list[str]): List of tissue names (e.g. ["bone", "soft"]).

    Returns:
        MAC matrix of shape (E_ENERGY, len(tissue_types)), where each column corresponds to one tissue.
    """
    MAC = xp.zeros((E_ENERGY, len(tissue_types)), dtype=float)

    for i, tissue in enumerate(tissue_types):
        mac_data = getattr(macs, tissue)
        if mac_data.size == 0:
            print(f"[MAC] Warning: No data found for '{tissue}', filling with zeros.")
            continue
        MAC[:, i] = mac_data[:E_ENERGY]

    return MAC

# @clean_gpu
def energyProjection(projections: Any,  # (W, H, ANGLES, T) #
                     voxel_size: tuple[float, float, float],
                     I0: Union[float, int],
                     spectrum: Any,
                     mac_matrix: Any,  # (E_ENERGY, T) 
                     I0_map: Optional[Any]= None) -> Any:
    """
    Apply energy-dependent projection.

    Parameters:
    - projections: xp.ndarray of shape (W, H, ANGLES, T), density units per tissue.
    - voxel_size: voxel size in mm (tuple).
 

    Returns:
    - proj_BH: xp.ndarray of shape (W, H, ANGLES, 1) raw xrays 

    Note:
    channel tissue order must be the same 
    """

    pixel_size = voxel_size[0] * 0.1  # convert mm to cm

    W, H, n_angles, n_tissues = projections.shape
    n_energies = len(spectrum)

    # Ensure macs shape is (E_ENERGY, T)
    macs = xp.asarray(mac_matrix)
    if macs.shape != (n_energies, n_tissues):
        raise ValueError(f"macs shape {macs.shape} does not match (E_ENERGY, T) = ({n_energies}, {n_tissues})")

    proj_BH = xp.zeros((W, H, n_angles), dtype=projections.dtype)

    # Reshape spectrum for proper broadcasting: (n_energies,) -> (1, 1, n_energies)
    spectrum = xp.asarray(spectrum).reshape(1, 1, n_energies)

    for i in range(n_angles):
        d_i = projections[:, :, i, :]  # (W, H, T) for angle i 
        # Expand dims for broadcasting:
        # d_i: (W, H, T) -> (W, H, T, 1)
        # macs_T_E: (T, E_ENERGY) -> (1, 1, T, E_ENERGY)
        # Result shape: (W, H, T, E_ENERGY)
        mu= d_i[:, :, :, None] * macs.T[None, None, :, :] * pixel_size

        # (W, H, T, E_ENERGY) -> (W, H, E_ENERGY)
        mu_sum = xp.sum(mu, axis=2)

        # Apply Beer-Lambert law and integrate over energies
        # spectrum: (1, 1, n_energies), exp(-mu_sum): (W, H, n_energies)
        # Result after broadcasting and sum: (W, H)
        if I0_map is None:
            I0_i = I0  # scalar
        else:
            I0_i = I0_map[:, :, i] # matrix (W,H)
                
        proj_BH_i = I0_i * xp.sum(spectrum * xp.exp(-mu_sum), axis=2)
        proj_BH[:, :, i] = proj_BH_i

    return proj_BH

def get_distance_map(geometry: Geometry)-> Any:
    """
    if tomo geometry
    complete 3d euclidean distance from source (0,0,zs) to detector pixel (x, SDD,z)
    xp.sqrt(x_grid**2 + SDD**2 + dz**2)
    """
    if isinstance(geometry,TomoGeometry):
        W, H = geometry.W, geometry.H
        SDD = geometry.SDD
        nprojs = geometry.nprojs
        z_step = geometry.step_mm


        distance_map = xp.zeros((H, W, nprojs))

        # Vertical positions of source: line along z axis
        z_positions = xp.linspace(-z_step * nprojs/2, z_step*nprojs/2, nprojs)

        pixel_size = geometry.pixel_size* geometry.binning_proj

        x = (xp.arange(W) - W / 2) * pixel_size[0]
        z = (xp.arange(H) - H / 2) * pixel_size[1]
        x_grid, z_grid = xp.meshgrid(x, z)

        for i, z_s in enumerate(z_positions):
            dz = z_grid - z_s     # z difference between pixel and source
            distance_map[...,i] = xp.sqrt(x_grid**2 + SDD**2 + dz**2)
    else:
        pass

    return xp.swapaxes(distance_map, 0,1)


def get_Poisson_component(
        projections: Any # raw projection or volume wirh I0=I,
    ):                     
    return xp.random.poisson(lam=projections)
  
def get_Gaussian_component(
        shape, 
        mu_dark: float,  
        sigma_dark: float,
        ):                     
    return xp.random.normal(mu_dark, sigma_dark, shape)
  
        

    


