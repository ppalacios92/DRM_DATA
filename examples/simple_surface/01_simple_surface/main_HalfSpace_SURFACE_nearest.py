# === Importar ShakerMaker ===
from shakermaker import shakermaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
# from shakermaker.stf_extensions.discrete import Discrete
from shakermaker.stf_extensions.gaussian import Gaussian
from shakermaker.slw_extensions import DRMHDF5StationListWriter
from shakermaker.sl_extensions import DRMBox
from shakermaker.sl_extensions.SurfaceGrid import SurfaceGrid

from shakermaker.sl_extensions import PointCloudDRMReceiver
import numpy as np

# # ---------------------------------------------------------------
# # Initialize CrustModel
# crustal = CrustModel(4)    
# vp,vs,rho,thick,Qa,Qb = 1.32, 0.75, 2.4000, 0.200, 1000., 1000.
# crustal.add_layer(thick, vp, vs, rho, Qa, Qb)
# vp,vs,rho,thick,Qa,Qb = 2.75, 1.57, 2.5000, 0.800, 1000., 1000.
# crustal.add_layer(thick, vp, vs, rho, Qa, Qb)
# vp,vs,rho,thick,Qa,Qb = 5.50, 3.14, 2.5000, 14.50, 1000., 1000.
# crustal.add_layer(thick, vp, vs, rho, Qa, Qb)
# vp,vs,rho,thick,Qa,Qb = 7.00, 4.00, 2.6700, 0.000, 1000., 1000.
# crustal.add_layer(thick, vp, vs, rho, Qa, Qb)

#Initialize CrustModel
crust = CrustModel(2)

#Slow layer
vp=4.000
vs=2.000
rho=2.600
Qa=10000.
Qb=10000.
thickness = 1.0

crust.add_layer(thickness, vp, vs, rho, Qa, Qb)

#Halfspace
vp=6.000
vs=3.464
rho=2.700
Qa=10000.
Qb=10000.
thickness = 0   #Infinite thickness!
crust.add_layer(thickness, vp, vs, rho, Qa, Qb)


M0 = 1e18/5e14/2
sigma = 0.06
t0 = 6*sigma
               
# ---------------------------------------------------------------
# Create source
z = 2.0                 # Source depth (km)
s, d, r = 0., 90., 0.   # Fault plane angles (deg)
source = PointSource([0, 0, z], 
                     [s, d, r],
                     stf = Gaussian(t0=t0, freq=1/sigma, M0=M0 , derivative=False))

fault = FaultSource([source], metadata={"name": "LOH1_source"})


x_station_DRM = [6.0, 8.0, 0.0]

# # ---------------------------------------------------------------
# # DRM Box Specification
# Lx = 110/1000  # km
# Ly = 110/1000  # km
# Lz = 35/1000  # km
# dx = 10/1000  # km
# nx, ny, nz = int(Lx/dx), int(Ly/dx), int(Lz/dx)
# # -----------------------------------------------------------------
# stations = DRMBox(x_station_DRM,
#                   [nx, ny, nz],
#                   [dx, dx, dx],
#                   metadata={"name": "01_DRM"})
# folder = '01_DRM'
# # ---------------------------------------------------------------





# # -----------------------------------------------------------------
# # XY plane at z=0km
# Lx = 10  # km
# Ly = 10  # km
# Lz = 0.030  # km
# dx = 100/1000  # km
# nx, ny, nz = int(Lx/dx), int(Ly/dx), int(Lz/dx)
# stations = SurfaceGrid([4,4,0],
#                        [nx, ny, nz],
#                        [dx, dx, dx],
#                        mode='plane',
#                        plane_z=0.0,
#                        metadata={"name": "02_Plane_Z_0"})
# folder = '02_Plane_Z_0'
# # ---------------------------------------------------------------





# # -----------------------------------------------------------------
# # XZ plane at y=8km
# Lx = 1  # km
# Ly = 1  # km
# Lz = 500/1000  # km
# dx = 50/1000  # km
# nx, ny, nz = int(Lx/dx), int(Ly/dx), int(Lz/dx)
# stations = SurfaceGrid(x_station_DRM,
#                        [nx, ny, nz],
#                        [dx, dx, dx],
#                        mode='plane',
#                        plane_y=8.0,
#                        metadata={"name": "03_Surface_xz_y8"})

# folder = '03_Surface_xz_y8'
# # -----------------------------------------------------------------





# # -----------------------------------------------------------------
# # YZ plane at x=6km
# Lx = 1  # km
# Ly = 1  # km
# Lz = 500/1000  # km
# dx = 50/1000  # km
# nx, ny, nz = int(Lx/dx), int(Ly/dx), int(Lz/dx)
# stations = SurfaceGrid(x_station_DRM,
#                        [nx, ny, nz],
#                        [dx, dx, dx],
#                        mode='plane',
#                        plane_x=6.0,
#                        metadata={"name": "04_Surface_yz_x6"})
# folder = '04_Surface_yz_x6'
# # -----------------------------------------------------------------





# # -----------------------------------------------------------------
# # Hollow box (DRM boundary)
# Lx = 110/1000  # km
# Ly = 110/1000  # km
# Lz = 35/1000  # km
# dx = 5/1000  # km
# nx, ny, nz = int(Lx/dx), int(Ly/dx), int(Lz/dx)

# stations = SurfaceGrid(x_station_DRM,
#                        [nx, ny, nz],
#                        [dx, dx, dx],
#                        mode='hollow',
#                        metadata={"name": "05_HollowBox"})
# folder = '05_HollowBox'

# # -----------------------------------------------------------------





# # -----------------------------------------------------------------
# # Full 3D grid
# Lx = 110/1000  # km
# Ly = 110/1000  # km
# Lz = 35/1000  # km
# dx = 5/1000  # km
# nx, ny, nz = int(Lx/dx), int(Ly/dx), int(Lz/dx)

# stations = SurfaceGrid(x_station_DRM,
#                        [nx, ny, nz],
#                        [dx, dx, dx],
#                        mode='filled',
#                        metadata={"name": "06_FilledBox"})
# folder = '06_FilledBox'
# # -----------------------------------------------------------------





# # -----------------------------------------------------------------
# # drm_nodes
# dx=1
# stations = PointCloudDRMReceiver(
#     point_cloud_file = './07_PointCloud_DRM/_drm_nodes.txt',
#     crd_scale        = 1/1e6,          # mm -> km
#     x0_fem           = [22000., 15500., 0.],   # origen FEM (ajusta si es diferente)
#     drmbox_x0        = x_station_DRM,
#     metadata         = {"name": "PointCloud_DRM"})
# folder = '07_PointCloud_DRM'
# # -----------------------------------------------------------------






# AI Attemps

# -----------------------------------------------------------------
# Hollow box (DRM boundary)
Lx = 110/1000  # km
Ly = 110/1000  # km
Lz = 35/1000  # km
dx = 10/1000  # km
nx, ny, nz = int(Lx/dx), int(Ly/dx), int(Lz/dx)

stations = SurfaceGrid(x_station_DRM,
                       [nx, ny, nz],
                       [dx, dx, dx],
                       mode='hollow',
                       metadata={"name": "08_AI_HollowBox"})
folder = '08_AI_HollowBox'

# -----------------------------------------------------------------






# ------------------------------5/---------------------------------
# Create model
model = shakermaker.ShakerMaker(crust, fault, stations)

# ---------------------------------------------------------------
# Parameters

gf_databasename = f'./{folder}/gf_database_{dx*1000:.0f}m.h5'
h5drm_output = f'./{folder}/Surface_{dx*1000:.0f}m.h5drm'

# Core params
dt = 0.005
nfft = 2048*2
dk = 0.05
tb = 20

# Output time window
tmin = 0.
tmax = 20.

# Tolerancias para clustering de GF
#Units 
_m = 0.001/1e12
delta_h=2.5*_m
delta_v_rec=2.5*_m
delta_v_src=2.5*_m
npairs_max = 100000     # max pairs per batch

# Writer
writer = DRMHDF5StationListWriter(h5drm_output)


# # Activate Green's function saving on all stations
# for station in stations:
#     station.metadata['save_gf'] = True
# ---------------------------------------------------------------
# Run con pipeline OP (3 stages)
model.run_nearest(
    stage='all',
    # stage=2,
    h5_database_name=gf_databasename,
    # Stage 0 params
    delta_h=delta_h,
    delta_v_rec=delta_v_rec,
    delta_v_src=delta_v_src,
    npairs_max=npairs_max,
    # Core params
    dt=dt,
    nfft=nfft,
    dk=dk,
    tb=tb,
    # Stage 1 & 2 params
    smth=1,
    # sigma=2,
    # taper=0.9,
    # wc1=1,
    # wc2=2,
    # pmin=0,
    # pmax=1,
    # nx=1,
    # kc=15.0,
    # Stage 2 only
    # tmin=tmin,
    # tmax=tmax,
    writer=writer,
    writer_mode='progressive',
    # General
    verbose=False,
    debugMPI=False,
    showProgress=True,
)