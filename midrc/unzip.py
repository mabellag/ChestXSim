import os
import zipfile
import argparse

def unzip_recursive(input_path, output_path):
    """
    Recursively search for .zip files inside input_path,
    extract them under output_path preserving relative folder structure,
    and delete the original zip file.
    """

    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)

                # Compute relative path to preserve folder structure
                rel_path = os.path.relpath(root, input_path)
                target_dir = os.path.join(output_path, rel_path)

                # Folder where contents will be extracted
                extract_folder = os.path.join(target_dir, os.path.splitext(file)[0])
                os.makedirs(extract_folder, exist_ok=True)

                try:
                    print(f"Decompressing: {zip_path}")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_folder)

                    print(f"Extracted to: {extract_folder}")

                    # OPTIONAL: delete zip
                    # os.remove(zip_path)
                    # print(f"Deleted zip: {zip_path}")

                except Exception as e:
                    print(f"Error decompressing {zip_path}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively unzip all ZIP files into output folder")
    parser.add_argument("input_folder", type=str, help="Folder containing ZIP files (recursively)")
    parser.add_argument("output_folder", type=str, help="Folder where files should be extracted")

    args = parser.parse_args()

    unzip_recursive(args.input_folder, args.output_folder)
    print("\nDone. All zip files processed.\n")

  
