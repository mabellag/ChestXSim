import numpy as np


# HLV_MAP: Half-value layer (HVL) data from spektr, expressed in millimeters.
# The keys are filter types as strings and the values are dictionaries mapping
# voltage (in kV) to the HVL value.
HLV_MAP = {  
    "1.5": {60: 3.1891, 80: 4.2038, 100: 4.4289, 110: 5.6343, 120: 6.0806},
    "2.5": {60: 3.4637, 80: 4.5624, 100: 5.5923, 110: 6.073, 120: 6.5346},
    "3.0":   {60: 3.5858, 80: 4.7302, 100: 5.7831, 110: 6.279, 120: 6.7406},
}

# https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z13.html
# # AL_ENERGY_MEV: Energy levels (in MeV) corresponding to the MAC values in AL_MAC_VAL.
# These values are from NIST and indicate the energy dependence of mass attenuation
AL_ENERGY_MEV = [ 
   1.00000E-03, 1.50000E-03, 1.55960E-03, 1.55960E-03, 2.00000E-03,
   3.00000E-03, 4.00000E-03, 5.00000E-03, 6.00000E-03, 8.00000E-03,
   1.00000E-02, 1.50000E-02, 2.00000E-02, 3.00000E-02, 4.00000E-02,
   5.00000E-02, 6.00000E-02, 8.00000E-02, 1.00000E-01, 1.50000E-01,
   2.00000E-01, 3.00000E-01, 4.00000E-01, 5.00000E-01, 6.00000E-01,
   8.00000E-01, 1.00000E+00, 1.25000E+00, 1.50000E+00, 2.00000E+00,
   3.00000E+00, 4.00000E+00, 5.00000E+00, 6.00000E+00, 8.00000E+00,
   1.00000E+01, 1.50000E+01, 2.00000E+01
]

# AL_MAC_VAL: Mass attenuation coefficients (MAC) in cm²/g, measured at the corresponding
# energy levels in AL_ENERGY_MEV
AL_MAC_VAL= [
   1.185E+03, 4.022E+02, 3.621E+02, 3.957E+03, 2.263E+03,
   7.880E+02, 3.605E+02, 1.934E+02, 1.153E+02, 5.033E+01,
   2.623E+01, 7.955E+00, 3.441E+00, 1.128E+00, 5.685E-01,
   3.681E-01, 2.778E-01, 2.018E-01, 1.704E-01, 1.378E-01,
   1.223E-01, 1.042E-01, 9.276E-02, 8.445E-02, 7.802E-02,
   6.841E-02, 6.146E-02, 5.496E-02, 5.006E-02, 4.324E-02,
   3.541E-02, 3.106E-02, 2.836E-02, 2.655E-02, 2.437E-02,
   2.318E-02, 2.195E-02, 2.168E-02
]

def calculate_effective_mac(voltage: int, filter:str = "1.5") -> float:
   """Calculate effective MAC given filter and voltage"""
   AL_density = 2.7  # g/cm³
   hvl = HLV_MAP.get(filter, {}).get(voltage)
   if hvl is None:
       raise ValueError(f"No HVL value found for filter {filter} and voltage {voltage}")
   
   hvl_cm = hvl * 0.1  # Convert HVL from mm to cm
   effective_mu = np.log(2)/hvl_cm  # Now μ will be in cm⁻¹
   return effective_mu/AL_density  # Result in cm²/g


def interpolate_effective_energy(mac: float) -> float:
    """
    Interpolate the effective energy based on the provided MAC (mass attenuation coefficient)
    using linear interpolation with NIST aluminum data.

    Since MAC is energy-dependent, this function determines which two MAC values the given 
    input 'mac' falls between and interpolates between their corresponding energy levels.

    Args:
        mac (float): The mass attenuation coefficient for which to interpolate the energy.
    
    Returns:
        float: The interpolated effective energy in keV, or None if the MAC is out of range.
    """

    energies = np.array(AL_ENERGY_MEV)
    macs = np.array(AL_MAC_VAL)
    
    for i in range(len(macs)-1):
        # get where the value falls between two values from table
        if macs[i+1] <= mac <= macs[i]:
            mac1, mac2 = macs[i], macs[i+1]
            e1, e2 = energies[i], energies[i+1]
            # linear interpolation 
            # (e2 - e1) total energy difference
            # (mac - mac1) / (mac2 - mac1) to indicate relevant position of mac1 and mac2 
            e_eff = e1 + (e2 - e1) * (mac - mac1) / (mac2 - mac1)
            return e_eff * 1000 # return kvp
    return None