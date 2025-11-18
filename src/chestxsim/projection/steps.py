import copy 
from chestxsim.core import (
    volumeData, SourceSpectrum, MACRepo, create_geometry_from_id,GEOMETRY_ID )
from chestxsim.wrappers.astra import *
from chestxsim.utility.ops_utils import apply_channelwise
from . import functional as F
from typing import Any 

__all__ = [
    "Projection", 
    "PhysicsEffect", 
    "NoiseEffect"
]

def ensure_4d(volume: Any) -> Any:
    """ Converts 3D volume (H,W,D) to 4D volume (H,W,D,T) where T=1 """
    if volume.ndim == 3:
        volume = volume[..., xp.newaxis]  # add tissue dimension at the end
    return volume

class Projection:
    def __init__(self, opt:Astra_OP, channel_wise:bool=True):
        self.opt = opt
        self.channel_wise= channel_wise
    
    def __call__(self, ct_data:volumeData) -> volumeData:
        volume = ensure_4d(ct_data.volume)
        metadata = copy.deepcopy(ct_data.metadata)
        if not self.channel_wise:
            volume = xp.sum(volume, axis=-1,keepdims=True)

        processed_volume = F.project(volume, self.opt, metadata.voxel_size)  
        metadata.dim = processed_volume.shape
        metadata.voxel_size = [self.opt.geometry.pxW, self.opt.geometry.pxH]
        metadata.step_outputs[self.__class__.__name__] = {
            "kernel": type(self.opt).__name__, 
            "id": next(
                (name for name, cls in GEOMETRY_ID.items()
                if isinstance(self.opt.geometry, cls)),
                None
        ),
            "geometry": self.opt.geometry.to_dict(),
        }
        return  volumeData(volume=processed_volume, metadata=metadata)
    

class PhysicsEffect:
    """
    Simulates X-ray spectrum effects on x-ray projections using density units 
    it includes optional inverse square law scaling and logarithmic flood correction.

    Parameters:
        physics (Physics) - required:
            The X-ray source model containing spectrum, energy, and intensity information.

        apply_ISL (bool) - optional, default=True:
            If True, applies the inverse square law (ISL) based on geometry to simulate
            the spatial attenuation of X-ray intensity.

        geometry (Geometry) - optional:
            Required only if apply_ISL is True. Defines source-detector distances
            used for inverse square law scaling.

        apply_flood_correction (bool) - optional, default=False:
            If True, applies flood correction by normalizing the projection against a reference
            flood field (either provided or simulated).

        flood (ndarray or compatible) - optional:
            A user-supplied flood field. If not provided and correction is enabled,
            a simulated flood will be generated based on geometry and I₀.

        log (bool, optional): 
            Default is True. If True, applies logarithmic transformation to the projection.
    """
    
  

    def __init__(self, 
                 source: SourceSpectrum, 
                 ISL: bool = False,
                 apply_flood_correction: bool = False, 
                 flood: Optional[Any] = None,
                 log:bool= False):

        self.source = source
        self.apply_ISL = ISL
        self.apply_flood_correction = apply_flood_correction
        self.flood = flood
        self.log = log

    def __call__(self, ct_data: volumeData) -> volumeData:
       
        volume = ct_data.volume
        metadata = copy.deepcopy(ct_data.metadata)

        # Check for required metadata
        # if metadata.units != "density":
        #     raise ValueError("Metadata must specify 'units' (e.g., density).")

        tissue_types = metadata.step_outputs.get("TissueSegmenter", {}).get("tissue_types", [])
        if len(tissue_types) != volume.shape[-1]:
            raise ValueError("Mismatch between tissue types and volume channels.")
        
        # print here:
        for i, tissue in enumerate(tissue_types):
            print(f"[PhysicsEffect] Tissue: {tissue} at channel {i}")

        # Get energy-dependent mass attenuation coefficients
        _macRepo = MACRepo()
        mac_matrix= F.build_mac_matrix(len(self.source.spectrum), tissue_types, _macRepo)

        # Apply ISL-based scaling if required
        if self.apply_ISL:
            # get distance map from geometry 
            # if no projection key say user to add geometry to projection metadata 
            # geometry= TomoGeometry.from_dict(metadata.step_outputs["Projection"].get("geometry"))
            geometry = create_geometry_from_id(
                metadata.step_outputs["Projection"].get("id"),
                **metadata.step_outputs["Projection"].get("geometry")
            )
            
            distance_map = F.get_distance_map(geometry)
            I0_map =  self.source.I0 * (geometry.SDD / distance_map)**2
            processed_volume = F.energyProjection(
                volume,
                metadata.find("ct_vx"),
                self.source.I0,
                self.source.spectrum,
                mac_matrix,
                I0_map
            )
        else:
            processed_volume = F.energyProjection(
                volume,
                metadata.find("ct_vx"),
                self.source.I0,
                self.source.spectrum,
                mac_matrix,
            )
          

        # Apply flood correction
        if self.apply_flood_correction:
            if self.flood is not None:
                # chequear si el flood es un volume o una imagen 
                flood = self.flood
            elif self.flood is None and self.apply_ISL:
                flood = I0_map
            else:
                flood = xp.ones_like(processed_volume) * self.source.I0
            # processed_volume = xp.maximum((processed_volume / flood), 1 / flood)
            processed_volume = processed_volume / flood


        if self.log:
            processed_volume = -xp.log(processed_volume)


        # Update metadata
        metadata.dim = processed_volume.shape
        metadata.step_outputs[self.__class__.__name__] ={
            "spectrum": "Polychromatic" if self.source.poly_flag else "Monochromatic",
            "ISL": self.apply_ISL,
            "I0": self.source.I0,
            "flood_corrected": self.apply_flood_correction,
            "log_applied": self.log
        }

        return volumeData(volume=processed_volume, metadata=metadata)
 
class NoiseEffect: # just in case want to add noise to just geometric projections per tissue - 
    def __init__(self, 
                mu_dark: Optional[float]= None,
                sigma_dark: Optional[float]= None,
                dark_img: Optional[Any]= None,
                inhomgeneities_map: Optional[Any]= None,
                apply_flood_correction:bool=False,
                flood: Optional[Any] = None,
                log: bool= False
                ):
        
        self.mu_dark = mu_dark if mu_dark else 0.0
        self.sigma_dark = sigma_dark if sigma_dark else 0.0 
        self.dark_img = dark_img
        self.inhomgeneities_map = inhomgeneities_map
        self.apply_flood_correction = apply_flood_correction
        self.flood = flood
        self.log = log


    def __call__(self, ct_data:volumeData) -> volumeData:
        volume = ct_data.volume
        metadata = copy.deepcopy(ct_data.metadata)
        processed_volume = self._apply(volume, metadata)
        # Update metadata
        metadata.step_outputs[self.__class__.__name__] =({
            "mu_dark": self.mu_dark,
            "sigma_dark":  self.sigma_dark,
            "flood_corrected": self.apply_flood_correction,
            "log_applied": self.log,
            "inhomogeneities": True if self.inhomgeneities_map is not None else False
        
        })

        return volumeData(volume=processed_volume, metadata=metadata)

    def _apply(self, volume, metadata):

        poisson= F.get_Poisson_component(volume)
      
        if self.mu_dark is not None and self.sigma_dark is not None:
            gaussian = F.get_Gaussian_component((volume.shape[0],volume.shape[1]),
                                                 self.mu_dark, self.sigma_dark)
        
        elif self.dark_img is not None:
            gaussian = self.dark_img
        
        else:
            gaussian = xp.zeros((volume.shape[0], volume.shape[1]))
        
        if self.inhomgeneities_map is not None:
            inhomgeneities_map = self.inhomgeneities_map

        else:
            inhomgeneities_map = xp.ones((volume.shape[0],volume.shape[1]))
          
        processed_volume = poisson * inhomgeneities_map[:,:,None] + gaussian[:,:,None]
        # print("Poisson stats:", poisson.mean(), poisson.std())
        # print("Gaussian stats:", gaussian.mean(), gaussian.std())

        if self.apply_flood_correction:
            step_outputs = getattr(metadata, "step_outputs", {}) or {}
            physics_done = "PhysicsEffect" in step_outputs

            if not physics_done:
                # Skip flood correction if no physics info 
                print("[NoiseEffect] Skipping flood correction: PhysicsEffect not found.")
            else:
                last_key = list(step_outputs.keys())[-1]
                phys_data = step_outputs["PhysicsEffect"]

                if self.flood is not None:
                    flood = self.flood[:, :, None]

                elif phys_data.get("ISL", False):
                    geometry_info = step_outputs.get("Projection", {}).get("geometry")
                    if geometry_info is not None:
                        geometry = create_geometry_from_id(
                            metadata.step_outputs["Projection"].get("id"),
                            **geometry_info,
                        )
                        distance_map = F.get_distance_map(geometry)
                        I0_map = phys_data["I0"] / (distance_map ** 2)
                        flood = I0_map + gaussian[:, :, None]
                    else:
                        flood = xp.full_like(processed_volume, phys_data["I0"]) + gaussian[:, :, None]

                else:
                    flood = xp.full_like(processed_volume, phys_data["I0"]) + gaussian[:, :, None]

                mu_dark = xp.zeros_like(processed_volume) * (self.mu_dark or 0.0)
                processed_volume = (processed_volume - mu_dark) / (flood - mu_dark)
        
        if self.log:
            processed_volume = -xp.log(processed_volume)
        
        return processed_volume 
    
        
    
        
        
        
        

