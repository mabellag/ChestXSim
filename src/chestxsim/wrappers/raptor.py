import os
import subprocess 
import warnings
import numpy as np
from chestxsim.device_utils import xp
import re
from chestxsim.io_utils.save_manager import SaveManager
from chestxsim.io_utils.data_readers import RawReader
from chestxsim.data_containers import CBCTGeometry
from typing import Optional, List
from chestxsim.io_utils.save_manager import *

class Raptor_OP:
    """Raptor backprojection operator. Only supports adjoint (backprojection) operation."""

    def __init__(self, executable_path: str, geometry:CBCTGeometry, **kwargs ):

        """ kwargs are to configure the reconstruction according """
        self.path_fx = executable_path          # Path to the Raptor executable
        self.geometry = geometry                # CBCTGeometry object with scan parameters

        # # Optional Raptor command-line parameters
        # self.nprojs_file = kwargs.get("nprojs_file", 1)         # -p number of projections per file
        # self.nfiles = kwargs.get("nfiles", 1)                   # -n number of input files
        # self.npositions = kwargs.get("npositions", 1)           # -b number of bed positions
        # self.binning_reco = kwargs.get("binning_reco", 1)       # -j binning factor for reconstruction
        # self.log = kwargs.get("log", 0)                         # -l apply logarithmic transformation
        # self.init_angle = kwargs.get("init_angle", 1)           # -i initial angle for reconstruction
        # self.parker = kwargs.get("parker", 0)                   # -pw apply Parker weighting
        # self.flip = kwargs.get("flip", [0, 0, 0])               # -ro flip in [x, y, z] axes
        # self.voltage = kwargs.get("voltage", 0)                 # -v X-ray tube voltage
        # self.sp = kwargs.get("sp", 360)                           # -sp span angle in degrees

        # # Output control
        # self.roi = kwargs.get("roi", None)                      # -r ROI dimensions [x, y, z]
        # self.vx_size = kwargs.get("vx_size", None)              # voxel size [vx, vy, vz] in mm (informational)

        # Internal use
        self.sv = SaveManager(".")                              # SaveManager for temporary file handling

    
    def adjoint_operator(self, projections: Any, 
                         reco_dim: Optional[List[int]]= None, #roi
                         vx_size: Optional[List[int]]= None,
                         **kwargs):
        """
        Applies the Raptor backprojection operator (Aᵗp).
        here kwargs are: HU, cupping correction
        """
        HU = kwargs.get("HU", [0, None])

        
        # Creating temporary files to temporarily store the results
        tmp_path_in = "tmp_in_0_0.ctf"
        tmp_path_out = "tmp_out_HU.img" if HU[0] else "tmp_out.img"
        self.sv.save_volume(projections, Path("."), tmp_path_in )

        # Executing the backprojection command with fuxim
        proj_size = projections.shape[:3]
        command = self._generate_cmd_line_BP(
            path_in=tmp_path_in[:-8],  # remove '_0_0.ctf' suffix
            path_out=tmp_path_out,
            proj_size=proj_size,
            roi = reco_dim, 
            vx_size = vx_size,
            **kwargs,
        )

        self._execute_command_BP(command)
        if reco_dim is None:
            reco_dim = self.parse_volume_size_from_txt("./info_reco.txt")

        if reco_dim is None:
            vx_size = self.parse_voxel_size_from_txt("./info_reco.txt")

        result = self._load_result_BP(tmp_path_out, reco_dim)

        # clean
        os.remove(tmp_path_in)
        os.remove(tmp_path_out)

        # update variables when using default values computed in raptor 
        self.reco_dim = reco_dim
        self.vx_size = vx_size

        return result

    # def _generate_cmd_line_BP(self, path_in, path_out, proj_size, HU, roi, vx_size, cupping_correction_path):
    def _generate_cmd_line_BP(self, path_in, path_out, proj_size, roi, vx_size, **kwargs):
      
        HU = kwargs.get("HU", [0, None])
        cupping_path = kwargs.get("cupping_correction_path", None)
        hu_flag = HU[0]
        hu_value = HU[1] if hu_flag else None

        def get_kw(name, default):
            return kwargs.get(name, default)

        cmd = [
            self.path_fx,
            '-m', path_in,
            '-fo', path_out,
            '-#', '1',
            '-k', str(0 if hu_flag else 1),
            '-n', str(get_kw("nfiles", 1)),
            '-b', str(get_kw("npositions", 1)),
            # '-a', str(get_kw("nprojs_file", 1)),
            # '-p', str(get_kw("nprojs_file", 1)),
            '-a', str(proj_size[2]),
            '-p', str(proj_size[2]),
            '-d', f'{proj_size[0]} {proj_size[1]}',
            '-do', str(self.geometry.DOD),
            '-q', str(self.geometry.SDD - self.geometry.DOD),
            '-i', str(get_kw("init_angle", 0)),
            '-l', str(get_kw("log", 0)),
            '-sp', str(get_kw("sp", 360)),
            '-fi', 'artifacts 3 3',
            '-j', ' '.join([str(get_kw("binning_reco", 1))]*3),
            '-pc', 'ring 50',
            '-pw', str(get_kw("parker", 0)),
            '-ro', ' '.join(map(str, get_kw("flip", [0, 0, 0]))),
            '-v', str(get_kw("voltage", 0))
            ]

        if hu_flag:
            cmd += ['-h', str(hu_value)]
        
        if roi:
            cmd += ['-r', f'{roi[0]} {roi[1]} {roi[2]}']
        
        if vx_size:
            cmd += ['-c', f'{vx_size[0]} {vx_size[1]} {vx_size[2]}']

        if cupping_path:
            cmd += ['-bh', '0', cupping_path]


        return ' '.join(cmd) + '"'

    def _execute_command_BP(self, cmd: str):
        print(cmd)
        try:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("command sent to RapTor")
        except subprocess.CalledProcessError as e:
            warnings.warn(f"Projection command failed! Error code: {e.returncode}\nStdout:\n{e.stdout.decode()}\nStderr:\n{e.stderr.decode()}")

    def _load_result_BP(self, path_out, shape):
        return RawReader.read_volume(path_out, shape)

    def parse_volume_size_from_txt(self, path: str):
        with open(path, 'r') as file:
            file_content = file.read()

        pattern_x = r'reco_size_x = (\d+)'
        pattern_y = r'reco_size_y = (\d+)'
        pattern_z = r'reco_size_z = (\d+)'

        match_x = re.search(pattern_x, file_content)
        match_y = re.search(pattern_y, file_content)
        match_z = re.search(pattern_z, file_content)

        return [int(match_x.group(1)), int(match_y.group(1)), int(match_z.group(1))]
    
    def parse_voxel_size_from_txt(self, path):
        with open(path, 'r') as file:
            file_content = file.read()

        pattern_x = r'reco_pitch_x = ([\d.]+)'
        pattern_y = r'reco_pitch_y = ([\d.]+)'
        pattern_z = r'reco_pitch_z = ([\d.]+)'

        match_x = re.search(pattern_x, file_content)
        match_y = re.search(pattern_y, file_content)
        match_z = re.search(pattern_z, file_content)

    
        return [float(match_x.group(1)), float(match_y.group(1)), float(match_z.group(1))]

