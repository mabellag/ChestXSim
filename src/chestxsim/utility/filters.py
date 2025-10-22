from chestxsim.core import xp
from typing import Any
# import numpy.fft as fft


# TAKE INTO ACCOUNT WHEN SELECTING THE AXIS::
# check this::
# axis 0 : apply calculation "column-wise"
# axis 1: apply calculation "row-wise"


def ramp_filter(
                vol: Any, 
                axis: int = 1,
                max_freq: float = 0.5,
                offset_filter: float = 0.005,
                padding: bool = True, 
                mode: str = "reflect"):
        
        """
        Parameters:
        vol -- Input volume array
        axis -- Axis along which to apply filter 
        max_freq -- Maximum frequency cutoff 
        offset_filter -- Small offset to reduce noise 
        padding -- Whether to pad to power of 2 for FFT efficiency 
        mode -- Padding mode if padding enabled 
        """
        
        orig_shape = vol.shape
        axis_size = orig_shape[axis]
        
        if padding:
            # Find the smallest power of 2 that is greater than or equal to detector width
            pad_size = int(2**xp.ceil(xp.log2(axis_size)))
            pad_before = (pad_size - axis_size) // 2
            pad_after = pad_size - axis_size - pad_before
            
            # Pad along the specified axis
            pad_width = [(0, 0)] * len(orig_shape)
            pad_width[axis] = (pad_before, pad_after)
            vol = xp.pad(vol, pad_width, mode=mode)
        
        # Apply FFT along the specified axis
        vol_ft = xp.fft.fft(vol, axis=axis)
            
        # Create frequency array for the specified axis
        freq = (vol.shape[axis] - 1) * xp.linspace(-1/2, 1/2, vol.shape[axis])

        # Ramp filter implementation and broadcasting 
        ramp_filter = xp.fft.fftshift(xp.abs(2 * max_freq * freq / vol.shape[axis]) + offset_filter)
        filter_shape = [1] * len(vol.shape)
        filter_shape[axis] = vol.shape[axis]
        ramp_filter = ramp_filter.reshape(filter_shape)
            
        # Apply filter
        filt_ft = vol_ft * ramp_filter
        
        # Inverse FT
        filt_vol = xp.real(xp.fft.ifft(filt_ft, axis=axis))
        
        # Remove padding if applied
        if padding:
            # Create slices for each dimension
            slices = [slice(None)] * len(orig_shape)
            slices[axis] = slice(pad_before, pad_before + axis_size)
            
            filt_vol = filt_vol[tuple(slices)]
            
        return filt_vol