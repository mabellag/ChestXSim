# MIDRC Downloader

A simple Windows tool to download files listed in a **MIDRC manifest** using the official **Gen3** client.

## Contents of the package

Make sure you have **all these files in the same folder**:

- `MIDRC_Downloader.exe` → The graphical launcher (this program).  
- `gen3-client.exe` → The official Gen3 client (downloaded from [cdis-data-client releases](https://github.com/uc-cdis/cdis-data-client/releases)).  
- `MIDRC_manifest.json` → You can replace it with any other manifest you want to download.  
- `LICENSE` → Apache 2.0 license for the Gen3-client launcher.

## Prerequisites

- **Windows 10/11** (64-bit)  
- **Internet connection**  
- A valid **MIDRC API Key** (`credentials.json`), which you can generate in your MIDRC profile:  
  [https://data.midrc.org/identity](https://data.midrc.org/identity)  
  *(Keys expire every 30 days and can be regenerated.)*

## How to use

1. Generate your API Key on the MIDRC website and save the `credentials.json` file in an accessible folder.  
2. **Double-click** `MIDRC_Downloader.exe`.  
3. When prompted:
   - Select your `credentials.json` file.
   - If a `MIDRC_manifest.json` is present in the same folder, you can use it or choose a different manifest.
   - Choose the destination folder where the MIDRC files will be saved.
4. The program will:
   - Configure the `midrc` profile in Gen3.
   - **Display a message when the download starts** and **another when it finishes**.
   - Open a new console window to run the download and show the progress of each file.  
     *(Depending on the manifest size, this may take minutes or hours.)*
5. When finished, the downloaded files will be located in the chosen destination folder.

## Notes

- If you reuse the program and your API Key has expired, simply generate a new `credentials.json`.  

## License

This launcher is distributed under the terms of the **Apache License 2.0**.  
See the included `LICENSE` file for details.

## Credits

- Gen3 official client © 2025 The Linux Foundation (Apache 2.0).  
- The `.exe` was built with [PyInstaller](https://www.pyinstaller.org/) and is released under **Apache 2.0**.  
- **This launcher was developed by the BiiG research group at University Carlos III of Madrid (UC3M).**
