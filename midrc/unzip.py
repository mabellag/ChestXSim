"""
This script extracts MIDRC ZIP files *in place* and reorganizes them into a
shorter, directory structure.

Original MIDRC downloads typically follow this structure:

    <ROOT>/
        302028-000112/                            <- SUBJECT / PATIENT
            1.2.826.0.1.3680043.10.474....144033/ <- SeriesInstanceUID (long UID)
                1.2.826.0.1.3680043....144038.zip <- ZIP with DICOM files
                1.2.826.0.1.3680043....145158.zip <- ZIP with DICOM files
                ...
                 
 Creates a shortened directory structure under the same root:

       <ROOT>/<PATIENT>/nt_<series_suffix>/nt_<zip_suffix>/

For every DICOM file, renames it to:

       nt_<suffix>.dcm

   where <suffix> is the last block of the original filename (without extension),
   e.g. `1.2.826.0.1.3680043.10.474.302028.144037` -> `nt_144037.dcm`

All DICOMs end up directly in:

       <ROOT>/<PATIENT>/nt_<series_suffix>/nt_<zip_suffix>/

It deletes:
   - the original ZIP file
   - any now-empty intermediate UID folders (but keeps the patient folder).

The original full UIDs can always be recovered from the MIDRC manifest.
"""

import os
import zipfile
import argparse

PREFIX = "nt_"


def build_target_dir(root: str, file: str, input_path: str):
    rel_path = os.path.relpath(root, input_path)
    parts = rel_path.split(os.sep) if rel_path != "." else []

    # Patient folder
    patient_id = parts[0] if parts else "unknown_subject"
    patient_dir = os.path.join(input_path, patient_id)

    # Series suffix from directory name
    if len(parts) >= 2:
        full_series_uid = parts[-1]
        series_suffix = full_series_uid.split(".")[-1]
    else:
        series_suffix = "unknown"

    series_dir = os.path.join(patient_dir, f"{PREFIX}{series_suffix}")

    # Suffix from ZIP name
    zip_base = os.path.splitext(file)[0]
    zip_suffix = zip_base.split(".")[-1]

    zip_dir = os.path.join(series_dir, f"{PREFIX}{zip_suffix}")
    return zip_dir, patient_dir


def cleanup_empty_dirs(path: str, stop_dir: str) -> None:
    path = os.path.abspath(path)
    stop_dir = os.path.abspath(stop_dir)

    cur = path
    while True:
        if cur == stop_dir:
            break

        try:
            os.rmdir(cur)  # succeeds only if directory is empty
        except OSError:
            # Directory not empty or cannot be removed; stop
            break

        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            break
        cur = parent


def flatten_and_rename_dicoms(folder: str) -> None:
    folder = os.path.abspath(folder)

    # First pass: move + rename all DICOMs to the root of `folder`
    for dirpath, _, filenames in os.walk(folder):
        for fname in filenames:
            src = os.path.join(dirpath, fname)
            if not os.path.isfile(src):
                continue

            base, ext = os.path.splitext(fname)

            # Consider .dcm or no-extension as DICOM; skip others
            if ext and ext.lower() != ".dcm":
                continue

            suffix = base.split(".")[-1]
            new_name = f"{PREFIX}{suffix}.dcm"
            dest = os.path.join(folder, new_name)

            if src == dest:
                continue

            # Overwrite if exists (no counters)
            if os.path.exists(dest):
                os.remove(dest)

            os.replace(src, dest)

    # Second pass: remove all subdirectories now that files are at root
    for dirpath, dirnames, _ in os.walk(folder, topdown=False):
        for d in dirnames:
            subdir = os.path.join(dirpath, d)
            try:
                os.rmdir(subdir)
            except OSError:
                # Not empty (shouldn't happen after moving), ignore
                pass


def unzip_recursive(input_path: str) -> None:
    input_path = os.path.abspath(input_path)

    for root, _, files in os.walk(input_path):
        for file in files:
            if not file.endswith(".zip"):
                continue

            zip_path = os.path.join(root, file)

            # Determine where to place this ZIP's contents
            target_dir, patient_dir = build_target_dir(root, file, input_path)
            os.makedirs(target_dir, exist_ok=True)

            try:
                print(f"Decompressing: {zip_path}")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(target_dir)

                # Flatten any internal folders and rename DICOMs
                flatten_and_rename_dicoms(target_dir)
                print(f"Extracted and renamed files in: {target_dir}")

                # Delete the original ZIP
                os.remove(zip_path)
                print(f"Deleted ZIP: {zip_path}")

                # Remove now-empty UID folders up to the patient folder
                cleanup_empty_dirs(root, patient_dir)

            except Exception as e:
                print(f"Error decompressing {zip_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Recursively extract MIDRC ZIP files IN PLACE, creating folders "
            "<PATIENT>/nt_<series_suffix>/nt_<zip_suffix>/ and renaming DICOM "
            "files to nt_<last_digits>.dcm (no extra UID subfolders)."
        )
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Root MIDRC folder (e.g., path containing patient subfolders).",
    )

    args = parser.parse_args()

    unzip_recursive(args.input_folder)
    print("\nDone. All ZIP files processed.\n")
