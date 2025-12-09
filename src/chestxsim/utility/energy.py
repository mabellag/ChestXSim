"""
Utility functions to compute the **effective energy** (in keV) of an X-ray beam
based on its tube voltage and aluminum filtration.

The method implemented here follows the standard HVL-based approximation:

1. A precomputed **Half-Value Layer (HVL)** lookup table (`HLV_MAP`) provides
   HVL values in millimeters of aluminum for several filtration levels
   ("1.5 mm", "2.5 mm", "3.0 mm") and tube voltages between **60–120 kVp**.

2. From the HVL, the script computes the corresponding **linear attenuation
   coefficient** μ using:
       μ = ln(2) / HVL

3. Dividing μ by the density of aluminum (2.7 g/cm³) yields an **effective
   mass attenuation coefficient** (MAC) in cm²/g.

4. The effective MAC is then **inverted** by linear interpolation against
   reference aluminum mass-attenuation data from NIST (`AL_ENERGY_MEV`,
   `AL_MAC_VAL`) to estimate the **effective monoenergetic energy** of the beam.

This allows approximating the beam’s spectral hardness using only:
    - Tube voltage (kVp)
    - Aluminum filtration category ("1.5", "2.5", "3.0")

**Supported voltages:**  
    60, 80, 100, 110, 120 kVp  
(Values outside this range are not available in the HVL table)
"""


import numpy as np
# HLV_MAP: Half-value layer (HVL) data from spektr, expressed in millimeters.
# The keys are filter types as strings and the values are dictionaries mapping
# voltage (in kV) to the HVL value.
HLV_MAP = {  
    "1.5": {60: 3.1891, 80: 4.2038, 100: 4.4289, 110: 5.6343, 120: 6.0806},
    "2.5": {60: 3.4637, 80: 4.5624, 100: 5.5923, 110: 6.073, 120: 6.5346},
    "3.0":  {60: 3.5858, 80: 4.7302, 100: 5.7831, 110: 6.279, 120: 6.7406},
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