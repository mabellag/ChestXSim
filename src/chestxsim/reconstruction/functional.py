
from chestxsim.core import xp, Geometry
from chestxsim.wrappers.astra import Astra_OP
# from chestxsim.wrappers.fuxsim import FuXSim_OP
# from chestxsim.wrappers.raptor import Raptor_OP
from typing import Union, Optional, List, Tuple
from chestxsim.utility.filters import *


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


def resolve_reco_dim(reco_dim_mm, reco_dim_px, reco_vx, geometry=None):
    # 1) explicit px
    if reco_dim_px is not None:
        return tuple(int(v) for v in reco_dim_px)

    # 2) explicit mm
    if reco_dim_mm is not None and reco_vx is not None:
        return tuple(int(reco_dim_mm[i] / reco_vx[i]) for i in range(3))

    # 3) geometry FOV (fallback)
    if geometry is not None:
        if reco_vx is None:
            raise ValueError("resolve_reco_dim: reco_vx required when using geometry FOV.")
        fov_mm = geometry.fov()
        return tuple(int(fov_mm[i] / reco_vx[i]) for i in range(3))

    return None


@apply_channelwise
def backProject(opt,
            projections:Any, 
            reco_dim:Tuple[int, int, int],
            reco_vx_size:Tuple[float, float, float],  # reco voxel size
            ):
    return opt.backproject(projections,reco_dim, reco_vx_size)

@apply_channelwise
def sample_volume(volume, nprojs, sampling_rate):
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
    """Apply filter to projections, currently only ramp is supported."""
    if filter_type == "ramp":
        return ramp_filter(projections, axis, max_freq, offset_filter, padding)
    else:
        raise NotImplementedError(f"Filter '{filter_type}' not implemented.")

@apply_channelwise
def fdk(projections: Any,
        opt,
        reco_dim: Optional[Tuple[int, int, int]] = None,
        reco_vx_size: Optional[Tuple[float, float, float]] = None,
        filter_type: Optional[str] = "ramp",
        offset_filter: float = 0.05,
        axis: int = 0,
        max_freq: float = 0.5,
        padding: Optional[bool] = True,
        ) -> Any:
    """
    FDK reconstruction -builtin
    
    Automatically applies filtering for Astra. 
    Skips filtering for FuXSim and Raptor which apply filtering by  configuring parameters .
    """

    if isinstance(opt, Astra_OP):
        filtered = apply_filter(projections,
                                filter_type=filter_type,
                                offset_filter=offset_filter,
                                axis=axis,
                                max_freq=max_freq,
                                padding=padding)
    else:
        # For FuXSim and Raptor, projections are assumed pre-filtered
        filtered = projections

    return opt.backproject(filtered, reco_dim, reco_vx_size)


# def sart(opt, 
#          projections: Any,
#          reco_dim: Tuple[int, int, int],
#          reco_vx_size: Tuple[float, float, float],
#          input_vol_size: Tuple[float, float, float],
#          input_vx_size:Tuple[float, float, float],
#          lamb = 1, 
#          n_iter = 20, 
#          eps = 1e-10,
#          x_0 = None):
    
#     r"""Applies SART algorithm to reconstruct the an image 
    
#     SART aims to solve

#     .. math::
#         out = x = \max_{x \in \mathbb{R}^N}} \|Ax - p\|^2_W

#     where A is a FUXIM linear (projection) operator defined by F_OP.forward_operator, x is the target image, 
#     and p = meas are the projections. SART solves this by applying a sort of preconditioned gradient descent 

#     .. math::
#         x^{(k+1)} = x^{(k)} - \lambda D A^\top W (Ax^{(k)} - p ) ,
    
#     where :math:`\lambda` is the step size of the gradient descent
    
#     .. math::
#         D = \text{diag}(\frac{1}{A^\top 1}), and W = \text{diag}(\frac{1}{A1})

#     Args:
#         F_OP : Fuxim Operator. Must have implemented forward_operator and adjoint_operator properly
#         meas : Numpy array containing the raw measurements
#         lamb : Step size of the SART algorithm
#         n_iter : Maximum number of iterations for SART Algorithm.
#         eps : Stopping criterion. When the maximum correction is below eps, then the algorithm stops
#         x_0: Initial iterate of the SART algorith 
#                 If ``None``, will initialize as a zero array of size F_OP.reco_dim         

#     Shape:
#         - Input: :math:`(F_{OP}.proj_size[0], F_{OP}.proj_size[1], N_{Proj})` or :math:`(F_{OP}.proj_size, N_{Proj})`.
#         - Output: :math:`F_{OP}.reco_dim`, 
        
#     Examples::

#         >>> opt =  FuXSim_Tomo()
#         >>> # Give opt the geometry parameters
#         >>> meas = opt.forward_operator(GT);
#         >>> f = opt.FDK(meas);
#         >>> lamb = 1;
#         >>> f_sart = SART(opt, meas, lamb, n_iter = 5, eps= 1e-10, x_0 = f)
#     """
#     if x_0 is None :
#         x = xp.zeros(reco_dim)
#     else : 
#         x = x_0

#     A1 = 1/opt.forward_operator(cp.ones(input_vol_size), input_vx_size)
#     A1 = cp.where(cp.isfinite(A1), A1, 0.0)

#     At1 = 1 / opt.adjoint_operator(cp.ones(projections.shape, dtype=cp.float32), reco_dim, reco_vx_size)
#     At1 = cp.where(cp.isfinite(At1), At1, 0.0)  # Replace inf/nan with 0
#     # Will return some inf values, give us a warning, we deal with the inf
#     # By dealing with the infs here, we get rid of all potential problems.
    

#     norm_err = 2 * eps * cp.ones(projections.shape, dtype=cp.float32)
#     iiter = 0

#     while iiter < n_iter and cp.amax(cp.abs(norm_err)).get() > eps:
#         norm_err = A1 * (opt.forward_operator(x, input_vx_size) - projections)
#         corr_norm = At1 * opt.adjoint_operator(norm_err, reco_dim, reco_vx_size)
#         print(f"[SART] Iter {iiter}: Mean correction={cp.mean(cp.abs(corr_norm)).get():.6f}")

#         x = x - lamb * corr_norm
#         iiter += 1

#     print(f"[SART] Done. Total iterations: {iiter}")
#     return x



