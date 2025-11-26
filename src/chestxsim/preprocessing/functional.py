
"""Low-level array ops used by preprocessing steps."""
from typing import Tuple, Union, List, Optional, Any
import os 
import cupy as cp
import torch.nn.functional 
from torchvision import transforms
from chestxsim.core import xp, ndi, Geometry
from chestxsim.utility.ops_utils  import apply_channelwise
from chestxsim.utility.energy_computation import calculate_effective_mac, interpolate_effective_energy
from skimage.morphology import ball


def get_stretcher_mask_dl(volume: Any, batch_size:int=1, model_path: str= None)-> Any:
    """
    Generate a stretcher mask using a deep learning (DL) model.

    Args:
        volume (Any): A 3D NumPy or CuPy array representing the input volume,
                      with shape [x, y, z].
        batch_size (int, optional): Number of axial slices (along the z-axis)
                                    to process in a single forward pass. Default is 1.

    Returns:
        Any: A binary 3D mask (uint8) with the same shape as `volume`,
             where stretcher regions are marked with 1 and all other areas with 0.
    """

    # Determine backend
    is_cupy = isinstance(volume, cp.ndarray)
    
    # load model 
    # model_path =  model_path or os.path.join("..", "materials", "models", model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()

    # ---- Quartile normalization on `volume` needed for the model to perform okay ----
    volume = volume.astype(xp.float32, copy=True)
    finite_min = xp.nanmin(volume)
    volume = xp.where(xp.isnan(volume), finite_min, volume)

    q1 = xp.percentile(volume, 25.0)
    q3 = xp.percentile(volume, 75.0)
    denom = float(q3 - q1)
    if denom == 0.0:
        vmin, vmax = float(xp.min(volume)), float(xp.max(volume))
        denom = (vmax - vmin) if (vmax - vmin) != 0.0 else 1.0
        volume = (volume - vmin) / denom
    else:
        volume = (volume - q1) / denom
        volume = xp.clip(volume, 0.0, 1.0)

    # ---- orientation swap BEFORE forward (legacy behavior) ----
    input_volume = xp.swapaxes(volume, 0, 1)  # [W, H, D]

    # setup transforms (deterministic bilinear + antialias)
    resize_transform  = lambda t: torch.nn.functional.interpolate(t, size=(512, 512), mode="bilinear",
                                                align_corners=False, antialias=True)
    inverse_resize    = lambda t: torch.nn.functional.interpolate(t, size=(input_volume.shape[1], input_volume.shape[0]),
                                                mode="bilinear", align_corners=False, antialias=True)
    sigmoid = torch.nn.Sigmoid()

    # make predictions
    binary_mask_vol = xp.zeros_like(volume, dtype=xp.uint8)  # final shape [H, W, D]
    with torch.no_grad():
        for i in range(0, input_volume.shape[2], batch_size):
            slices = input_volume[:, :, i:i + batch_size]                  # [W, H, B]
            slices = xp.transpose(slices, (2, 0, 1)).astype(xp.float32)    # [B, W, H]

            # CuPy/Numpy -> Torch
            if is_cupy:
                # tensor = torch.utils.dlpack.from_dlpack(slices)            # GPU tensor
                tensor = torch.from_dlpack(slices)
            else:
                tensor = torch.tensor(slices, dtype=torch.float32)

            tensor = tensor.unsqueeze(1).to(device)                         # [B, 1, W, H]
            tensor = resize_transform(tensor)                               # [B, 1, 512, 512]

            preds = sigmoid(model(tensor))                                  # [B, 1, 512, 512]
            preds = inverse_resize(preds)                                   # [B, 1, W, H]
            preds = (preds.squeeze(1) > 0.5).type(torch.uint8)             # [B, W, H]

            preds = preds.permute(2, 1, 0)                                  # back to [H, W, B]

            if is_cupy:
                preds_xp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(preds))
            else:
                preds_xp = preds.cpu().numpy()

            binary_mask_vol[:, :, i:i + batch_size] = preds_xp

    return binary_mask_vol
    
def get_stretcher_mask_analytical(volume: Any, threshold:int)->Any:
    """
    Generate a stretcher mask using thresholding, morphological operations, and connected component analysis (CCA).

    Args:
        volume (Any): A 3D NumPy or CuPy array representing the input volume,
                      with shape [x, y, z].
        threshold (int): Intensity threshold used to segment the patient from background air.

    Returns:
        Any: A binary 3D mask (uint8) with the same shape as `volume`,
             where stretcher regions are marked with 1 and other areas with 0.
    """
     
    _window_size = (4,4,4)
    _struct_size = 2 
    
    input_volume = ndi.median_filter(volume, size=_window_size)
    binary_mask_vol = xp.where(input_volume>= threshold, 1,0)
    binary_mask_vol = ndi.binary_opening(binary_mask_vol, structure= xp.asarray(ball(_struct_size)), iterations=1)

    # CCA - patient corresponds to the largest island 
    labeled_mask, _ = ndi.label(binary_mask_vol)
    region_sizes = xp.bincount(labeled_mask.ravel())
    region_sizes[0] = 0
    largest_region_label = region_sizes.argmax()
    largest_island = labeled_mask == largest_region_label

    # fill possible holes:
    filled = xp.zeros(largest_island.shape)
    for i in range(largest_island.shape[2]):
        filled[:,:,i] = ndi.binary_fill_holes(largest_island[:,:,i]) 

    return (1 - filled).astype(xp.uint8)

def remove_bed(volume: Any, mask: Any) -> Any:
    """
    Remove the stretcher from a ct volume using a binary mask.

    Args:
        volume (Any): A 3D NumPy or CuPy array representing the input volume,
                      with shape [x, y, z].
        mask (Any): A binary 3D mask (uint8) with the same shape as `volume`,
                    where stretcher regions are marked with 1 and other areas with 0.

    Returns:
        Any: A volume of the same shape as `volume`, where stretcher regions
             have been replaced with the minimum intensity value of the original volume.
    """
    masks_vol= xp.where(mask != 0, 1, 0 )
    return xp.where(masks_vol == 0, volume, volume.min())

def crop_air(volume: Any, axis: int = 1, tol: int = 5, delta: int = 5) -> Any:
    """
    Remove air regions from a 3D volume by cropping along a specified dimension.
    
    Args:
        volume: 3D NumPy or CuPy volume in [x, y, z] format.
        dim: Dimension to crop along (0 = y-axis, 1 = x-axis, 2 = z-axis, default: 1).
        tol: Tolerance threshold for determining air vs. non-air (default: 5).
        delta: Padding value added to detected boundaries (default: 5).
    
    Returns:
        cropped_volume: The 3D volume cropped along the specified dimension to exclude air regions.
        indx: A length-2 numpy array with the start and end indices along `dim` used for cropping.
              - indx[0] = starting slice index after adjusting for padding (delta)
              - indx[1] = ending slice index after adjusting for padding (delta)
              
    Example:
        If dim=1 (y-axis), and indx = [10, 50], then the returned volume will be
        volume[:, 10:50, :], i.e., only the slices from 10 to 49 along axis 1 are kept.
    """

    def get_air_index_dim(volume: Any, axis: int, tol: int, delta: int):
        P = xp.sum(volume, axis=tuple(d for d in range(3) if d != axis)) # if dim=0, produces tuple (1,2)
        P = P > tol # determines wich positions contain enough material to be non air 
        P = P.astype('float')
        P = P[1:] - P[:-1] # element wise difference - discrete difference 
        ind_init = xp.where(P == 1)[0] # transition between air (0) to material (1) 
        ind_end = xp.where(P == -1)[0] # transition bewtenen material(1) tpo air (0)
        
        ind_init = max(ind_init[0] - delta, 0) if ind_init.shape[0] > 0 else 0
        ind_end = min(ind_end[0] + delta, volume.shape[axis]) if ind_end.shape[0] > 0 else volume.shape[axis]
        
        return [int(ind_init), int(ind_end)]

    
    Aux = (volume - xp.amin(volume)) / (xp.amax(volume) - xp.amin(volume))
    indx = get_air_index_dim(Aux, axis, tol, delta)
    
    if axis == 0:
        volume = volume[indx[0]:indx[1], :, :]
    elif axis == 1: 
        volume = volume[:, indx[0]:indx[1], :]
    elif axis == 2:
        volume = volume[:, :, indx[0]:indx[1]]
    
    return volume, indx

def crop(volume: Any, values_mm:list[Union[int, float]], voxel_size: list[float], axis: int= 2) -> Any:
    """Crop 3D volume in mm across selected dimension
    
    Args:
        volume: 3D NumPy or CuPy volume in [x, y, z] format. 
        value_mm: Amount to crop in mm
        voxel_size: Voxel dimensions in mm [x,y,z]
        dim: Dimension to crop (default=2 for z-axis)
    
    Returns:
        Cropped volume
    """

    # Convert mm to voxels, ensure cropping indices are within bounds 
    start_crop = max(0, int(values_mm[0] / voxel_size[axis]))
    end_crop = max(0, int(values_mm[1] / voxel_size[axis]))

    # Create slice objects for each dimension
    slices = [slice(None)] * volume.ndim
    slices[axis] = slice(start_crop, end_crop)
    
    return volume[tuple(slices)]

def compute_extension_vals_from_geometry(volume: Any, voxel_size: List[float], geometry: Geometry, chest_center:Union[int,float]=150)-> List[float]:
    """
    compute extension values to fill full field of view of the geometry (including outer most ray positions which is ext values)

    ext' = P-L + ext where ext = tan(alpha) * D , alpha = atan((S-P)/ sdd)  in mm
    where 
    P is half detector height
    L is  patient height
    S is half source distance
    SDD source to detector
    D is patient depth (direction y)
    """
    _,_,target_z = geometry.fov() 
    # print(target_z)
    return compute_extension_vals_from_target_height(volume,voxel_size, target_z, chest_center)

def compute_extension_vals_from_target_height(volume: Any, voxel_size: List[float], target_height:Union[int,float] , chest_center:Union[int,float]=150)->List[float]:
    """ compute extention needed up and bottom in mm to reach extended volume lenght
    making sure detector center is aligned with chest center in the z axis
    """
    init_lenght =  volume.shape[2]*voxel_size[2]
    return  [round(val,2) for val in [target_height/2 - chest_center, target_height/2 - (init_lenght - chest_center)] ]

def extend_volume(volume: Any, ext_vals_mm: List[Union[int, float]], voxel_size: List[float]) -> Any:
    """Extend 3D volume in mm across the z axis
    
    Args:
        volume: CT volume
        ext_vals_mm: Extension values in mm [up, down]
        voxel_size: Voxel dimensions in mm [x,y,z]
    
    Returns:
        Extended volume in z direction (height)
    """
    # print(ext_vals_mm)
    ext_up_mm, ext_down_mm = ext_vals_mm
            
    # Convert mm to slices 
    ext_up = int(round(ext_up_mm / voxel_size[2]))
    ext_down = int(round(ext_down_mm / voxel_size[2]))

    # print(ext_up, ext_down)
    
    new_shape = (volume.shape[0], volume.shape[1], volume.shape[2] + ext_up + ext_down)
    extended_volume = xp.zeros(new_shape, dtype=volume.dtype)
    
    # Place the original volume in the middle
    extended_volume[:, :, ext_up:ext_up + volume.shape[2]] = volume
    
    # Extend upwards (at the beginning/top)
    if ext_up > 0:
        first_slice = volume[:, :, 0:1]
        for i in range(ext_up):
            extended_volume[:, :, i:i+1] = first_slice
    
    # Extend downwards (at the end/bottom)
    if ext_down > 0:
        last_slice = volume[:, :, -1:]
        for i in range(ext_down):
            idx = ext_up + volume.shape[2] + i
            extended_volume[:, :, idx:idx+1] = last_slice
    
    return extended_volume

def get_bone_mask_dl(volume:Any, batch_size:int=1, model_path:Optional[str]= None)-> Any:
    """
    Generate a bone mask from a 3D volume using a deep learning model.

    Args:
        volume (Any): A 3D NumPy or CuPy array in [x, y, z] format.
        batch_size (int, optional): Number of axial slices to process at once. Default is 1.

    Returns:
        Any: A binary mask of the same shape as `volume`, in [x, y, z] format, dtype=uint8.
             Bone regions are labeled with 1, background with 0.
    """
    # Determine backend
    is_cupy = isinstance(volume, cp.ndarray)

    # load model
    model_path = model_path or os.path.join("..", "materials", "models", "model_fine_tune_vf_2.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()

    # nomalization used to train the model
    input_volume = xp.swapaxes(volume, 1,0)  # [y, x, z] -> [x, y, z]
    hu_min =  -1024
    hu_max = 2000 
    input_volume = xp.clip((input_volume - hu_min ) / (hu_max - hu_min), 0,1)
    input_volume = input_volume.astype(xp.float32)

    # setup transforms 
    resize_transform = transforms.Resize((512, 512))
    inverse_resize = transforms.Resize((input_volume.shape[0], input_volume.shape[1]))
    sigmoid = torch.nn.Sigmoid()

    # make predictions 
    binary_mask_vol = xp.zeros_like(input_volume, dtype=xp.uint8) 
    with torch.no_grad():
        for i in range(0, input_volume.shape[2], batch_size):
            # start_time = time.time()
            slices = input_volume[:, :, i:i + batch_size] # shape: [H, W, B]
            slices = xp.transpose(slices, (2, 0, 1)).astype(xp.float32) # shape: [B, H, W]
            tensor = torch.tensor(slices, dtype=torch.float32).unsqueeze(1).to(device) # shape: [B, 1, H, W]

            # Convert to PyTorch tensor
            if is_cupy:
                # CuPy -> DLPack -> PyTorch (GPU)
                tensor = torch.utils.dlpack.from_dlpack(slices)
            else:
                tensor = torch.tensor(slices, dtype=torch.float32)

            tensor = tensor.unsqueeze(1).to(device)  # [B, 1, H, W]
            tensor = resize_transform(tensor)
            preds = sigmoid(model(tensor))
            preds = inverse_resize(preds)
            preds = (preds.squeeze(1) > 0.5).type(torch.uint8) # [B, H, W]
            preds = preds.permute(1, 2, 0) #[H, W, B]

            # Convert preds back to correct backend
            if is_cupy:
                preds_xp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(preds))
            else:
                preds_xp = preds.cpu().numpy()

            # Assign to volume
            binary_mask_vol[:, :, i:i + batch_size] = preds_xp
    
    return xp.swapaxes(binary_mask_vol,0,1)

def get_bone_mask_analytical(volume:Any, bone_threshold: int, )-> Any:
    """
    Generate a bone mask from a 3D volume using thresholding and morphological operations.

    Args:
        volume (Any): A 3D NumPy or CuPy array in [x, y, z] format.
        bone_threshold (int): HU threshold above which voxels are considered bone.

    Returns:
        Any: A binary mask of the same shape as `volume`, in [x, y, z] format, dtype=uint8.
             Bone regions are labeled with 1, background with 0.
    """
    _window_size = (4,4,4)
    _struct_size = 2
    volume = ndi.median_filter(volume, size= _window_size)
    bone_mask = xp.where(volume > bone_threshold, 1, 0)
    bone_mask = ndi.binary_closing(bone_mask, structure=xp.asarray(ball(_struct_size)), iterations=1)
    return bone_mask.astype(xp.int8)

def segment_volume(volume: Any, tissue_masks: Any) -> Any:
    """
    Segment a 3D volume into tissue-specific maps using provided binary masks.

    Args:
        volume (Any): A 3D NumPy or CuPy array in [x, y, z] format with background and bed removed.
        tissue_masks (Any): A 4D array of shape [x, y, z, T], where T is the number of tissue types.

    Returns:
        Any: A stacked volume of shape [x, y, z, T], dtype matches volume. 
             Each channel corresponds to one tissue type.
    """
    tissue_map_list = [
        xp.where(tissue_masks[..., i] == 1, volume, volume.min())
        for i in range(tissue_masks.shape[-1])
    ]
    return xp.stack(tissue_map_list, axis=-1)  # shape: [x, y, z, T]

def compute_effective_energy(voltage: int, filter: str = "1.5") -> int:
    """
    Compute the effective energy (in keV) based on the specified voltage and filter.

    This function implements a two-step process:
    1. It calculates the effective mass attenuation coefficient (MAC) for aluminum using the
       provided voltage and filter setting by calling `calculate_effective_mac`.
       - The calculation is based on the half-value layer (HVL) data from spektr, which is stored in HLV_MAP.
       - The HVL (in mm) is converted to cm, and then the effective MAC is derived using the formula:
             effective_mu = ln(2) / (HVL in cm)
       - mac value is compute dividing the effective_mu by  aluminum density (2.7 g/cm³).
    2. It interpolates the effective energy (in keV) from the computed MAC using the NIST AL data by calling
       - The NIST data arrays for energies and MAC values (AL_ENERGY_MEV and AL_MAC_VAL) are used to determine
         the effective energy through linear interpolation.

    Args:
        voltage (int): The voltage (in kV) for which to compute the effective energy.
        filter (str): The filter setting as a string. Default is "1.5"

    Returns:
        int: The computed effective energy in keV.
    """
    mac_al_eff = calculate_effective_mac(voltage, filter)
    return interpolate_effective_energy(mac_al_eff)

def convert_HU_to_mu(volume: Any, mu_water: float)-> Any:
    """
    Convert a CT volume in HU units to a volume of linear attenuation coefficients (mu).

    This conversion is based on the formula:
        mu = mu_water * (1 + HU / 1000)
    where mu_water is the attenuation coefficient of water at effective energy
    
    Negative values (if any) are set to zero.

    Args:
        volume (xp.ndarray): CT volume in HU. 3D NumPy or CuPy volume in [x, y, z]
        mu_water (float): Attenuation coefficient for water at effective energy.
    
    Returns:
        xp.ndarray: CT volume converted to linear attenuation coefficients.
    """
    volume = mu_water * (1 + volume / 1000)
    return xp.clip(volume, 0, volume.max())
 
def convert_HU_to_density(tissue_volume: Any, tissue_mac_eff: float, water_mac_eff: float, mu_factor:int=1)-> Any:
    """
    Args:
        volume:  3D CT volume in HU units  NumPy or CuPy volume in [x, y, z] segmented tissue
        mac_eff (float): Attenuation coefficient for water at effective enrgy.
        factor: factor to twick the density if needed  
    """
    volume_mu = mu_factor * convert_HU_to_mu(tissue_volume, water_mac_eff)
    volume_d = xp.where(volume_mu > 0,  volume_mu/ tissue_mac_eff, 0) 
    return volume_d

@apply_channelwise
def flip(volume: Any, axis:int=1)->Any:
    return xp.flip(volume,axis)

@apply_channelwise
def rotate(volume: Any, angle: float, axes=(1, 0), mode="nearest", order=0) -> Any:
    """
    Apply rotation to a 3D volume using SciPy's `ndi.rotate`.

    Args:
        volume (Any): 3D NumPy or CuPy array in (x, y, z) format.
        angle (float): Rotation angle in degrees (counter-clockwise).
        axes (tuple): The plane of rotation. Default is (1, 0) = (y, x).
        mode (str): Points outside boundaries filled according to this mode.
        order (int): Interpolation order (0=nearest, 1=linear, etc).

    Returns:
        Rotated volume (same shape and type as input).
    """
    return ndi.rotate(volume, angle, axes=axes, mode=mode, order=order)

    