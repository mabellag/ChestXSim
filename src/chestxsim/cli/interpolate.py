"""
Interpolate CT volumes to match DCT reconstructed volume resolution 
obtained with match_input=True. 

The input should be:
    - CT with original dimensions (ct_dim, ct_vx in metadata)
    - Bed already removed (by default reading from results/CT_without_bed)
"""

import os
import argparse
from typing import Tuple

from chestxsim.io import RawReader, SaveManager, STEP_TO_FOLDER, RESULTS_DIR
from chestxsim.utility import Interpolator


def parse_args():
    parser = argparse.ArgumentParser(description="Resample CT to match DCT resolution")

    parser.add_argument(
        "--input", "-i",
        default=str(RESULTS_DIR / STEP_TO_FOLDER["BedRemover"]),
        help=f"Folder with CT without bed "
             f"(default: results/{STEP_TO_FOLDER['BedRemover']})"
    )

    parser.add_argument(
        "--vx_xyz", "-vx",
        default="1.25,5.0,1.25",
        help="Target voxel size for DCT reconstruction, e.g. '1.25,5.0,1.25'"
    )

    return parser.parse_args()



def run(input_folder: str,
        target_vx_size: Tuple[float, float, float] = (1.25, 5.0, 1.25)):

    for root, _, files in os.walk(input_folder):

        if not files:
            continue
        else:
            print(root)
            input_ct = RawReader().read(root)
        
            md = input_ct.metadata

            # Physical size of the input in mm
            ct_dim = md.find("ct_dim")          # (Nx, Ny, Nz)
            ct_vx  = md.find("ct_vx")           # (vx_x, vx_y, vx_z)

            input_mm = [ct_dim[i] * ct_vx[i] for i in range(3)]
            target_size = [int(round(input_mm[i] / target_vx_size[i])) for i in range(3)]

            print(f"\nResampling folder: {root}")
            print(f"  Original CT dim:  {ct_dim}")
            print(f"  Original vx size: {ct_vx}")
            print(f"  Physical size mm: {[round(x,2) for x in input_mm]}")
            print(f"  → Target dim:     {target_size}")
            print(f"  → Target vx size: {target_vx_size}")

            # Resample
            interpolator = Interpolator(target_vx_size, tuple(target_size))
            resampled_ct = interpolator(input_ct)

            SaveManager().save_step("Interpolator", resampled_ct)


def main():
    args = parse_args()
    target_vx = tuple(float(v.strip()) for v in args.vx_xyz.split(","))
    run(input_folder=args.input, target_vx_size=target_vx)


if __name__ == "__main__":
    main()


