# # Only import path constants here (no heavy modules!)
# from .paths import (
#     find_project_root,
#     PROJECT_ROOT,
#     MATERIALS_DIR,
#     EXECUTABLES_DIR,
#     MODELS_DIR,
#     MAC_DIR,
#     SPECTRUM_DIR,
#     RESULTS_DIR,
#     EXAMPLES_DIR,
#     INPUTS_DIR,
#     SETTINGS_DIR,
#     STEP_TO_FOLDER,
#     UNITS_TO_FOLDER,
#     TISSUE_TO_FOLDER,
#     SPECTRUM_TO_FOLDER,
# )

# __all__ = [
#     "find_project_root",
#     "PROJECT_ROOT",
#     "MATERIALS_DIR",
#     "EXECUTABLES_DIR",
#     "MODELS_DIR",
#     "MAC_DIR",
#     "SPECTRUM_DIR",
#     "RESULTS_DIR",
#     "EXAMPLES_DIR",
#     "INPUTS_DIR",
#     "SETTINGS_DIR",
#     "STEP_TO_FOLDER",
#     "UNITS_TO_FOLDER",
#     "TISSUE_TO_FOLDER",
#     "SPECTRUM_TO_FOLDER",
#     "SaveManager",
#     "CTReader",
#     "DicomReader",
#     "RawReader",
# ]


# def __getattr__(name):
#     if name == "SaveManager":
#         from .save_manager import SaveManager
#         return SaveManager
#     if name in {"CTReader", "DicomReader", "RawReader"}:
#         from .readers import CTReader, DicomReader, RawReader
#         return {"CTReader": CTReader, "DicomReader": DicomReader, "RawReader": RawReader}[name]
#     raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
