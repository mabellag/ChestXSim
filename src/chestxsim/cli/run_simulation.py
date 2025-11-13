
"""

ChestXsim Pipeline Runner 

This is the main script for running ChestXsim simulation pipeline. 
It allows processing chest CT data in DICOM or RAW formats through a configurable pipeline composed
of preprocesing, projection and/or reconstruction modules. 

Usage (from command line):
---------------------------
python main.py --input PATH_TO_CT_FOLDER --config CONFIG.json [--mode 0|1|2] [--output OUTPUT_DIR]

Arguments:
----------
--input / -i      : Path to folder containing CT data (DICOM or RAW format).
--config / -c     : Path to JSON configuration file describing pipeline steps and geometry.
--mode / -m       : Optional mode:
                    0 - Preprocessing only
                    1 - Projection only
                    2 - Reconstruction only
                    (Default: run full pipeline)
--output / -o     : Optional output directory to store results.

Dependencies:
------------
- ChestXsim mocules located in src/
- CuPy-compatible GPU for full acceleration.
- ASTRA Toolbox, FuXSim or Raptor executables



"""

import sys, os
# PROJECT_ROOT = r"D:\bhermosi\chestxsim-project" 
# SRC = os.path.join(PROJECT_ROOT, "src")
# if SRC not in sys.path:
#     sys.path.insert(0, SRC)

import time 
import os   
import argparse
from typing import Optional
import json
from chestxsim.core import volumeData, build_pipeline
from chestxsim.io import RawReader, DicomReader
from pathlib import Path


def load_config(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        return json.load(f)

def is_multitissue_root(path: Path) -> bool:
    return any((path / sub).is_dir() for sub in ["bone", "soft"])

def find_case_ids_across_tissues(base_path: Path) -> set:
    """Return unique set of case IDs found across tissue folders."""
    case_ids = set()
    for tissue_folder in base_path.iterdir():
        if not tissue_folder.is_dir():
            continue
        for case_dir in tissue_folder.iterdir():
            if case_dir.is_dir():
                case_ids.add(case_dir.name)
    return case_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Run CT processing pipeline.")
    parser.add_argument("--input", "-i", required=True, help="Path to input folder containing DICOM or RAW files")
    parser.add_argument("--config", "-c", required=True, help="Path to JSON config file")
    parser.add_argument("--mode", "-m", default=None, help=" Optional Execution mode: 0 (preprocessing), 1 (projection), or 2 (reconstruction)")
    parser.add_argument("--output", "-o", default=None, help="Path to output folder (optional)")
    return parser.parse_args()
    
def run(input_folder: str, config: str, mode: Optional[int] = None, output_folder: Optional[str] = None):
    """
    Execute the simulation pipeline on each CT volume found in the input folder.
    """    
    input_path = Path(input_folder)
    config_dict = load_config(config)
    pipeline = build_pipeline(config_dict, mode=int(mode) if mode is not None else None, output_folder=output_folder)
    print(f"Saving results to: {pipeline.save_manager.base_dir}")

    # MULTI-TISSUE INPUT CASE:
    if is_multitissue_root(input_path):
        print(f"Detected multi-tissue input structure under: {input_path}")
        reader = RawReader()
        case_ids = find_case_ids_across_tissues(input_path)

        for case_id in sorted(case_ids):
            print(f"Processing multi-tissue case: {case_id}")
            try:
                ct_data= reader.load_multi_tissue(input_path, case_id, combine_method="stack")
                start_time = time.time()
                result = pipeline.execute(ct_data)
                elapsed = time.time() - start_time
                print(f"{case_id} completed in {elapsed:.2f} seconds")
            except Exception as e:
                print(f"Skipping {case_id} due to error: {e}")
    else:
        # SINGLE VOLUME INPUT CASE:
        for root, _, files in os.walk(input_folder):
            # print to total of case id found here 
            if not files:
                continue

            is_raw = any(f.lower().endswith((".img", ".npy")) for f in files)

            if is_raw:
                reader = RawReader()
            else:
                reader = DicomReader(convert_to_HU=True, clip_values=(-1000, 3000))

            print(f"\nProcessing: {root}")
            ct_data = reader.read(root)

            start_time = time.time()
            result = pipeline.execute(ct_data)
            elapsed = time.time() - start_time

            case_id = result.metadata.id or os.path.basename(root)
            print(f"\n{case_id} completed in {elapsed:.2f} seconds")
        


def main(): 
    overall_t0 = time.time()
    args = parse_args()
    run(input_folder=args.input, config=args.config, mode=args.mode, output_folder=args.output)
    print(f"\nSimulation(s) completed in {time.time()-overall_t0:.2f}s.\n")

if __name__ == "__main__":
    main()
    