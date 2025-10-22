from chestxsim.core.device import xp
from typing import Any

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