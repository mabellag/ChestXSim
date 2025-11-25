
from chestxsim.core import xp, Geometry
from chestxsim.wrappers.astra import Astra_OP
from typing import Union, Optional, List, Tuple
from chestxsim.utility.filters import *


def resolve_reco_dim(reco_dim_mm, reco_dim_px, reco_vx, geometry=None):
    """
    Resolve the reconstruction volume dimensions using priority:
    (1) explicit pixel size, (2) explicit mm size, (3) geometry FOV
    """

    if reco_dim_px is not None:
        return tuple(int(v) for v in reco_dim_px)
    
    if reco_dim_mm is not None and reco_vx is not None:
        return tuple(int(reco_dim_mm[i] / reco_vx[i]) for i in range(3))

    if geometry is not None:
        if reco_vx is None:
            raise ValueError("resolve_reco_dim: reco_vx required when using geometry FOV.")
        fov_mm = geometry.fov()
        return tuple(int(fov_mm[i] / reco_vx[i]) for i in range(3))

    return None



def sample_volume(volume, nprojs, sampling_rate):
    """ downsample projection angles to simulate limited span angle acquisitions"""
    if nprojs*sampling_rate <= volume.shape[2]:
        projs = projs[:,:,:nprojs*sampling_rate]
        projs = projs[:, :, ::sampling_rate] 
        return projs 
    
def apply_filter(projections: Any,
                 filter_type: str = "ramp",
                 offset_filter: float = 0.05,
                 axis: int = 0,
                 max_freq: float = 0.5,
                 padding: bool = True,
                 **kwargs):
    """
    Apply a 1D frequency-domain filter to each projection.

    Args:
        projections (array): Input projection data.
        filter_type (str): Filter type, currently only "ramp".
        offset_filter (float): Low-frequency offset for numerical stability.
        axis (int): Axis along which to apply filtering.
        max_freq (float): Normalized frequency cutoff (0–0.5).
        padding (bool): Whether to pad before FFT.
    """
    if filter_type == "ramp":
        return ramp_filter(projections, axis, max_freq, offset_filter, padding)
    else:
        raise NotImplementedError(f"Filter '{filter_type}' not implemented.")


def fdk(opt,
        projections: Any,
        reco_dim: Optional[Tuple[int, int, int]] = None,
        reco_vx_size: Optional[Tuple[float, float, float]] = None,
        filter_type: Optional[str] = "ramp",
        offset_filter: float = 0.005,
        axis: int = 1,
        max_freq: float = 0.5,
        padding: Optional[bool] = True,
        ) -> Any:
    """
    FDK reconstruction.

    Applies a 1D filter to each projection, backprojects the result, and
    normalizes by the number of projections.

    Parameters
    ----------
    opt : object
        Operator with backproject().
    projections : array
        Projection data (nu, nv, n_proj).
    reco_dim : tuple (nx, ny, nz)
        Output volume size.
    reco_vx_size : tuple (sx, sy, sz)
        Output voxel size in mm.
    filter_type : str
        Projection filter ("ramp").
    offset_filter : float
        Small offset added to the filter.
    axis : int
        Detector axis to filter along.
    max_freq : float
        Maximum normalized frequency.
    padding : bool
        Whether to pad before FFT.

    Returns
    -------
    array
        Reconstructed volume (nx, ny, nz).
    """
 
    filtered = apply_filter(projections,
                                filter_type=filter_type,
                                offset_filter=offset_filter,
                                axis=axis,
                                max_freq=max_freq,
                                padding=padding)
    fdk = opt.backproject(filtered, reco_dim, reco_vx_size)
    norm_fdk = fdk/opt.geometry.nprojs
    return norm_fdk

def sart(opt, 
         projections: Any,
         reco_dim: Tuple[int, int, int],
         reco_vx_size: Tuple[float, float, float],
         lamb = 1, 
         n_iter = 20, 
         eps = 1e-10,
         x_0 = None):
    
    """Applies SART algorithm to reconstruct the an image
    it solves  
        x_{k+1} = x_k + λ * D * Aᵀ * W * (p - A x_k)

    where:
        - A is the projection operator (opt.project)
        - Aᵀ is its adjoint / backprojection (opt.backproject)
        - W = 1 / (A 1)    
        - D = 1 / (Aᵀ 1)  
        - λ is the relaxation parameter (step size)

    
    Parameters
    ----------
    opt : object
        Projection operator with methods:
            - opt.project(volume, vx_size)
            - opt.backproject(projections, reco_dim, reco_vx_size)
    projections : array-like
        Measured projections p.
    reco_dim : tuple of int
        Output volume dimensions (nx, ny, nz).
    reco_vx_size : tuple of float
        Voxel size of the reconstruction grid.
    input_vol_size : tuple of int
        Size of the input volume grid used to compute A1 via ones(input_vol_size).
    lamb : float
        Relaxation parameter (typically between 0.1 and 1.0).
    n_iter : int
        Maximum number of iterations.
    eps : float
        Stopping tolerance based on the normalized projection error.
    x_0 : array-like or None
        Initial reconstruction. If None, initialized to zeros on the reconstruction grid.

    Returns
    -------
    x : array
        The reconstructed volume on the reconstruction grid.
    """
    
    if x_0 is None :
        x = xp.zeros(reco_dim)
    else : 
        x = x_0

    A1 = 1/opt.project(xp.ones(reco_dim), reco_vx_size)
    A1 = xp.where(xp.isfinite(A1), A1, 0.0)
  
    At1 = 1 / opt.backproject(xp.ones(projections.shape, dtype=xp.float32), reco_dim, reco_vx_size)
    At1 = xp.where(xp.isfinite(At1), At1, 0.0)
   

    norm_err = 2 * eps * xp.ones(projections.shape, dtype=xp.float32)
    iiter = 0

    while iiter < n_iter and xp.amax(xp.abs(norm_err)).get() > eps:
      
        norm_err = A1 * (opt.project(x, reco_vx_size) - projections)
        corr_norm = At1 * opt.backproject(norm_err, reco_dim, reco_vx_size)
        print(f"[SART] Iter {iiter}: Mean correction={xp.mean(xp.abs(corr_norm)).get():.6f}")

        x = x - lamb * corr_norm
        iiter += 1

    print(f"[SART] Done. Total iterations: {iiter}")
    return x




