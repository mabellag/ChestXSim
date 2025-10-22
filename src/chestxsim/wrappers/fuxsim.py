from typing import List, Optional, Any
import subprocess
import warnings
import os 
import re, glob

from chestxsim.io_utils.save_manager import *
from chestxsim.io_utils.data_readers import *
from chestxsim.data_containers import TomoGeometry, CBCTGeometry
from chestxsim.device_utils import xp


def remove_file(path_file):
    """
    Deletes file given by path_file
    """
    os.remove(path_file)


class FuXSim_OP:
    r"""Fuxim simulation object to execute FuXIM projection and backprojection operations.
    it generates the projection back projection command line to execute the operations based on parameters  uses temporal files 
    base object 
    """
  
    def __init__(self, executable_path: str):
        self.path_fx: str = executable_path     # Path to the executable
        self.proj_size: List[int] = [0, 0]      # Detector size [-d]
        self.px_size: List[float] = [0.0, 0.0]  # Pixel size [-ip]
        self.SDD: float = 0.0                   # Source-Detector Distance [-sd]
        self.DOD: float = 0.0                   # Detector-Origin Distance [-do]
        self.nprojs: int = 0                    # Number of projections
        self.voltage: int = 0                   # Voltage used for projection

        self.sv = SaveManager(".")              # uses from io_utils to save raw images to pass in command line  

    def forward_operator(self, volume:Any, voxel_size:list[float]):
        r"""
        Applies the  projection operator
        We model the projection operation by a linear operation :math:`p = Ax + n`,
        where :math:`p \in \mathbb{R}^M` are the projections, :math:`A \in \mathbb{R}^{M\times N}`
        is the system matrix, :math:`x \in \mathbb{R}^x` is the ground truth, :math:`n \in \mathbb{R}^N`
        is a noise vector.
        This function computes :math:`Ax \in \mathbb{R}^M`


        Args:
            f : 3-D numpy array. Its size much be of self.reco_size.

        Outputs:
            m : 3-D numpy array of size (self.px_size, self.n_steps+1)

      

        """
        # Creating temporary files to temporarily store the results
        tmp_path_in = Path("tmp_in.img")
        tmp_path_out = Path("tmp_out.img")
        self.sv.save_volume(volume, Path("."), tmp_path_in )

        # Executing the projection command with fuxim
        self._execute_command_P(tmp_path_in, tmp_path_out, volume.shape, voxel_size)
        meas = self._load_result_P(tmp_path_out, 4)

        # print(meas.shape)

        # removing temporary files
        tmp_path_in= str(tmp_path_in)
        tmp_path_out= str(tmp_path_out)
    
        remove_file(tmp_path_in)
        # remove_file(tmp_path_in[:-4] + '.hdr')
        path_tmp = tmp_path_out[:-4] + "_{}x{}x{}".format(meas.shape[0], meas.shape[1], meas.shape[2]) + tmp_path_out[-4:]
        path_tmp_out = tmp_path_out[:-4] + "_{}x{}x{}".format(meas.shape[0], meas.shape[1], meas.shape[2]) + "_parameters.txt"
        remove_file(path_tmp_out)
        remove_file(path_tmp)
        
        return meas
    
    
    def adjoint_operator(self, projections: Any, 
                         reco_dim: Optional[List[int]]= None, 
                         vx_size: Optional[List[int]]= None,
                         **kwargs):
                     
        r"""
        Applies the back-projection operator
        We model the projection operation by a linear operation :math:`p = Ax + n`,
        where :math:`p \in \mathbb{R}^M` are the projections, :math:`A \in \mathbb{R}^{M\times N}`
        is the system matrix, :math:`x \in \mathbb{R}^x` is the ground truth, :math:`n \in \mathbb{R}^N`
        is a noise vector.
        This function computes :math:`A^T m \in \mathbb{R}^N`

        """
        

        # Creating temporary files to temporarily store the results
        tmp_path_in = Path("tmp_in.img")
        tmp_path_out = Path("tmp_out.img")
        self.sv.save_volume(projections, Path("."), tmp_path_in )

        # Executing the backprojection command with fuxim
        proj_size = projections.shape[:3]
        self._execute_command_BP(tmp_path_in, 
                                tmp_path_out, 
                                proj_size,
                                reco_dim, 
                                vx_size, 
                                **kwargs)
        
        if reco_dim is None:    
            # Look for any *_parameters.txt file matching the base name
            base = os.path.splitext(tmp_path_out)[0]  # removes ".img"
            matches = glob.glob(base + "*_parameters.txt")
            
            if not matches:
                raise FileNotFoundError("No *_parameters.txt file found matching output name.")

            path_to_parameters_txt = matches[0]  # Take the first match
            reco_dim = self.parse_volume_size_from_txt(path_to_parameters_txt)
            print(f"Using default value for -r (reconstruction roi){reco_dim}")
        
        if vx_size is None:
            # Look for any *_parameters.txt file matching the base name
            base = os.path.splitext(tmp_path_out)[0]  # removes ".img"
            matches = glob.glob(base + "*_parameters.txt")
            
            if not matches:
                raise FileNotFoundError("No *_parameters.txt file found matching output name.")

            path_to_parameters_txt = matches[0]  # Take the first match
            vx_size = self.parse_voxel_size_from_txt(path_to_parameters_txt)
            print(f"Using default value for -c (reconstruction vx size){vx_size}")
        
        
        # loading the backprojected data
        reco = self._load_result_BP(str(tmp_path_out), 4, reco_dim)

        # # removing temporary files
        tmp_path_in= str(tmp_path_in)
        tmp_path_out= str(tmp_path_out)
    
        remove_file(tmp_path_in)
        # path_tmp_1 = tmp_path_in[:-4] + '.hdr'
        path_tmp = tmp_path_out[:-4] + "_{}x{}x{}".format(reco_dim[0], reco_dim[1], reco_dim[2]) + tmp_path_out[-4:]
        path_tmp_2 = tmp_path_out[:-4] + "_{}x{}x{}".format(reco_dim[0], reco_dim[1], reco_dim[2]) + '_parameters.txt'
        
        # remove_file(path_tmp_1)
        remove_file(path_tmp)
        remove_file(path_tmp_2)

        self.reco_dim = reco_dim
        self.vx_size = vx_size
        return reco



    def _execute_command_P(self, path_in: str, path_out: str, vol_dim: List[int], vx_size: List[float]) -> None:
        cmd = self._generate_cmd_line_P(path_in, path_out, vol_dim, vx_size)

        try:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Projection command sent to FuXSim")
        except subprocess.CalledProcessError as e:
            warnings.warn(f"Projection command failed! Error code: {e.returncode}\nStdout:\n{e.stdout.decode()}\nStderr:\n{e.stderr.decode()}")

    def _execute_command_BP(self, path_in: str, path_out: str, proj_size, reco_dim: List[int], vx_size: List[float], **kwargs) -> None:
        cmd = self._generate_cmd_line_BP(path_in, path_out, proj_size, reco_dim, vx_size, **kwargs)
        try:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Backprojection command sent to FuXSim")
        except subprocess.CalledProcessError as e:
            warnings.warn(f"Backprojection command failed! Error code: {e.returncode}\nStdout:\n{e.stdout.decode()}\nStderr:\n{e.stderr.decode()}")
    
    def _load_result_P(self, path_out, ext_size):
        path_out = str(path_out)
        print(self.nprojs)
        shape = (self.proj_size[0], self.proj_size[1], self.nprojs)
        path_out_modif = path_out[:-ext_size] + "_{}x{}x{}".format(shape[0], shape[1], shape[2]) + path_out[-ext_size:]

        image = RawReader.read_volume(path_out_modif, shape)
        return image
    
    def _load_result_BP(self, path_out: str, ext_size: int, reco_dim: List[int]):
        path_out_modif = f"{path_out[:-ext_size]}_{reco_dim[0]}x{reco_dim[1]}x{reco_dim[2]}{path_out[-ext_size:]}"
        image = RawReader.read_volume(path_out_modif, reco_dim)
        return image

    def _generate_cmd_line_BP(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement _generate_cmd_line_BP")

    def _generate_cmd_line_P(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement _generate_cmd_line_P")
    def parse_volume_size_from_txt(self, path):
        with open(path, 'r') as f:
            content = f.read()

        pattern_x = r'Volumen size:\s*(\d+)\s'
        pattern_y = r'Volumen size:\s*\d+\s(\d+)\s'
        pattern_z = r'Volumen size:\s*\d+\s\d+\s(\d+)'

        match_x = re.search(pattern_x, content)
        match_y = re.search(pattern_y, content)
        match_z = re.search(pattern_z, content)

        if not (match_x and match_y and match_z):
            raise ValueError("Failed to extract volume size from parameters file.")

        return [int(match_x.group(1)), int(match_y.group(1)), int(match_z.group(1))]


    def parse_voxel_size_from_txt(self, path):
        """
        Parses the voxel size from a *_parameters.txt file.
        
        Returns:
            List of 3 floats: [vx_size_x, vx_size_y, vx_size_z]
        """
        with open(path, 'r') as file:
            content = file.read()

        pattern = r'Pixel size volume:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        match = re.search(pattern, content)

        if not match:
            raise ValueError("Could not find voxel size in file.")

        return [float(match.group(1)), float(match.group(2)), float(match.group(3))]



class FuXSim_Tomo(FuXSim_OP):
    r"""Tomosynthesis simulation object to execute FuXIM projection and backprojection operations.
    """

    def __init__(self, executable_path:str, geometry: Optional[TomoGeometry] = None):
        super().__init__(executable_path)
        
        if geometry is not None:
            self.configure_from_geometry(geometry)
        
        else:
            self.step = float = 1.0 
            self.bucky = float = 0.0

        self.nprojs = self.nprojs+1


    def _generate_cmd_line_BP(self, path_in: str, path_out: str, proj_size, reco_dim: List[int], vx_size: List[float],**kwargs) -> str:
        
        offset_filter = kwargs.get("offset_filter") #-ft [flag ofset axis]
        if offset_filter:
            filter = [1, offset_filter, 1]
        else:
            filter = [0,0,0]

        HU = kwargs.get("HU", [0, None])

        print(proj_size)
        # if HU is None:
        #     HU = []

        hu_flag = HU[0]
        hu_value = HU[1] if hu_flag else None

        detector_object_distance = self.bucky + reco_dim[1] * vx_size[1] / 2

        cmd = [
            self.path_fx,
            "-fi", str(path_in),
            "-fo", str(path_out),
            "-a", str(self.nprojs),
            "-m", "0", "3", "0", 
            "-#", "2",
            "-d", str(proj_size[0]), str(proj_size[1]),
            "-ip", str(self.px_size[0]), str(self.px_size[1]),
            "-do", str(detector_object_distance),
            "-sd", str(self.SDD),
            "-fp", "0",
            "-jo", str(self.step), "0",
            "-ft", str(filter[0]),"0", str(filter[1]), "1", 
            "-fm", str(filter[2]),
            "-vi", "2",
        ]

        if hu_flag:
            cmd += ['-h', str(hu_value)]
        
        if reco_dim:
            cmd += ['-r', f'{reco_dim[0]} {reco_dim[1]} {reco_dim[2]}']

        if vx_size:
            cmd += ['-c', f'{vx_size[0]} {vx_size[1]} {vx_size[2]}']

        cmd= " ".join(cmd)
        print(cmd)
        return cmd


    def _generate_cmd_line_P(self, path_in: str, path_out: str, vol_dim: List[int], vx_size: List[float]) -> str:
        """Generate forward projection command line.
        """
        # Calculate detector-to-origin distance with specific volume dimensions
        detector_object_distance = self.bucky + vol_dim[1] * vx_size[1] / 2
        
        cmd = (
            f'{self.path_fx} -fi {path_in} -fo {path_out}'
            f' -m 1 3 0 -# 2 -k 2'
            f' -b {vol_dim[0]} {vol_dim[1]} {vol_dim[2]}'
            f' -c {vx_size[0]} {vx_size[1]} {vx_size[2]}'
            f' -ip {self.px_size[0]} {self.px_size[1]}'
            f' -d {self.proj_size[0]} {self.proj_size[1]}'
            f' -do {detector_object_distance} -sd {self.SDD}'
            f' -fp 0 -a {self.nprojs}'
            f' -jo {self.step} 0 -vw 1 1'
        )
        
        print(cmd)
        return cmd

    
    def configure_from_dict(self, config:dict)-> "FuXSim_Tomo":
        self.proj_size = [config["geometry"]["detector_size"][0] // config["geometry"]["binning_proj"],
                         config["geometry"]["detector_size"][1] // config["geometry"]["binning_proj"]]
        self.px_size = [config["geometry"]["pixel_size"][0] * config["geometry"]["binning_proj"],
                       config["geometry"]["pixel_size"][1] * config["geometry"]["binning_proj"]]
        self.SDD = config["geometry"]["SDD"]
        self.bucky = config["geometry"]["bucky"]
        self.step = config["geometry"]["step"]
        self.nprojs = config["geometry"]["nsteps"]
        return self
    
    def configure_from_geometry(self, geometry: TomoGeometry)->"FuXSim_Tomo":
        self.geometry = geometry
        self.proj_size = [
            geometry.detector_size[0] // geometry.binning_proj,
            geometry.detector_size[1] // geometry.binning_proj
        ]
        self.px_size = [
            geometry.pixel_size[0] * geometry.binning_proj,
            geometry.pixel_size[1] * geometry.binning_proj
        ]
        
        # Set tomosynthesis-specific parameters
        self.SDD = geometry.SDD
        self.bucky = geometry.bucky
        self.step = geometry.step
        self.nprojs = geometry.nstep
        self.DOD: Optional[float] = None  # Will be calculated when needed
        self.angles: Any = xp.array([0])  # Single angle for tomo
        
        return self   


class FuXSim_CBCT(FuXSim_OP):
    r"""CBCT simulation object.

    Can be used to execute FuXIM projection and backprojection operations.
    It can either be used through paths (i.e. a path_in for a raw file is given, and through execute_command_P
    (resp. execute_command_BP), the projections (resp. backprojected image) are saved in path_out), or by inputing numpy arraysç
    with the methods forward_operator and adjoint_operator.
    """

    def __init__(self, executable_path:str, geometry: Optional[CBCTGeometry]= None):
        super().__init__(executable_path)
        if geometry is not None:
            self.configure_from_geometry(geometry)
        
        else:
            self.step_angle: float = 1.0  # Angular increment between projections in degrees
            self.init_angle: int = 0  # Initial angle in degrees

    def _generate_cmd_line_BP(self, 
                              path_in: str,
                              path_out: str, 
                              proj_size,
                              reco_dim, 
                              vx_size, 
                              **kwargs) -> str:
        """Generate backprojection command line.
        
        Args:
            path_in: Input projection path
            path_out: Output volume path
            reco_dim: Reconstruction dimensions [x, y, z]
            vx_size: Voxel size [x, y, z] in mm
            HU: Optional Hounsfield unit parameters
            
        Returns:
            Command line string
        """
        filter = kwargs.get("filter", [0, 0, 0])#-ft [flag ofset axis]
        sp = kwargs.get("sp", 360) 

        HU = kwargs.get("HU", [0, None])
        # if HU is None:
        #     HU = []
        hu_flag = HU[0]
        hu_value = HU[1] if hu_flag else None
            
        
        # Base command with common parameters
        cmd = [
            self.path_fx,
            "-fi", str(path_in),
            "-fo", str(path_out),
            "-a", str(proj_size[2]),
            "-m", "0", "0", "0",
            "-#", "2",
            "-d", str(proj_size[0]), str(proj_size[1]),
            "-ip", str(self.px_size[0]), str(self.px_size[1]),
            "-do", str(self.DOD),
            "-sd", str(self.SDD),
            "-sp", str(sp),
            "-i", str(self.init_angle),
            "-fp", "0",
            "-ft", str(filter[0]), "0", str(filter[1]), "1",
            "-fm", str(filter[2]),
            "-vi", "2"
        ]

        # Add Hounsfield unit parameters if available
        if HU and HU[0] == 1:
            cmd += ["-k", "0", "-h", str(HU[1]), "-v", str(self.voltage)]
        else:
            cmd += ["-k", "2"]

        if reco_dim:
            cmd += ['-r', f'{reco_dim[0]} {reco_dim[1]} {reco_dim[2]}']
        
        if vx_size:
            cmd += ['-c', f'{vx_size[0]} {vx_size[1]} {vx_size[2]}']
      
        cmd= " ".join(cmd)
        print(cmd)
        return cmd

    def _generate_cmd_line_P(self, path_in: str, path_out: str, vol_dim: List[int], vx_size: List[float]) -> str:
        """Generate forward projection command line.
        
        Args:
            path_in: Input volume path
            path_out: Output projection path
            vol_dim: Volume dimensions [x, y, z]
            vx_size: Voxel size [x, y, z] in mm
            
        Returns:
            Command line string
        """
        # Calculate scan angle
        scan_angle = self.step_angle * self.nprojs
        
        cmd = (
            f'cmd /c"{self.path_fx} -fi {path_in} -fo {path_out}'
            f' -m 1 0 0 -# 2 -k 2'
            f' -b {vol_dim[0]} {vol_dim[1]} {vol_dim[2]}'
            f' -c {vx_size[0]} {vx_size[1]} {vx_size[2]}'
            f' -ip {self.px_size[0]} {self.px_size[1]}'
            f' -d {self.proj_size[0]} {self.proj_size[1]}'
            f' -sp {scan_angle} -i {self.init_angle}'
            f' -do {self.DOD} -sd {self.SDD}'
            f' -fp 0 -a {self.nprojs}"'
        )
        
        print(cmd)
        return cmd

    def configure_from_dict(self, config: dict)-> "FuXSim_CBCT":
        self.proj_size = [config["geometry"]["detector_size"][0] // config["geometry"]["binning_proj"],
                         config["geometry"]["detector_size"][1] // config["geometry"]["binning_proj"]]
        self.px_size = [config["geometry"]["pixel_size"][0] * config["geometry"]["binning_proj"],
                       config["geometry"]["pixel_size"][1] * config["geometry"]["binning_proj"]]
        self.SDD = config["geometry"]["SDD"]
        self.DOD = config["geometry"]["DOD"]
        self.step_angle = config["geometry"]["step_angle"]
        self.init_angle = config["geometry"]["init_angle"]
        self.nprojs = config["geometry"]["nprojs"]
        return self
    
    def configure_from_geometry(self, geometry: CBCTGeometry) ->"FuXSim_CBCT":
        self.geometry = geometry
        self.proj_size = [
            geometry.detector_size[0] // geometry.binning_proj,
            geometry.detector_size[1] // geometry.binning_proj
        ]
        self.px_size = [
            geometry.pixel_size[0] * geometry.binning_proj,
            geometry.pixel_size[1] * geometry.binning_proj
        ]
        
        self.SDD = geometry.SDD
        self.DOD = geometry.DOD
        self.step_angle = geometry.step_angle
        self.init_angle = geometry.init_angle
        self.nprojs = geometry.nprojs
        return self 
