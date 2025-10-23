try:
    import cupy as cp
    from cupyx.scipy import ndimage as ndi
    
    if cp.cuda.is_available():
        xp = cp
        device_count = cp.cuda.runtime.getDeviceCount()
        current_device = cp.cuda.device.get_device_id()
        
        print(f"GPU detected: Using CuPy with {device_count} device(s), current device: {current_device}")
    else:
        import numpy as np
        from scipy import ndimage as ndi
        xp = np
        print("No GPU available: Using NumPy (CPU)")

except ImportError:
    import numpy as np
    from scipy import ndimage as ndi
    xp = np
    print("CuPy not installed: Using NumPy (CPU)")

except Exception as e:
    from scipy import ndimage as ndi
    import numpy as np
    xp = np
    print(f"GPU initialization failed: {e}. Using NumPy (CPU)")
