import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
from mpi4py import MPI
from shakermaker import shakermaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.ffspsource import FFSPSource
from shakermaker.stf_extensions.srf2 import SRF2
from shakermaker.slw_extensions import DRMHDF5StationListWriter
from shakermaker.sl_extensions import DRMBox
from shakermaker.sl_extensions.SurfaceGrid import SurfaceGrid

# -------------------------------------------------------------------------------
# MPI Setup
# -------------------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# -------------------------------------------------------------------------------
# Load FFSP source from HDF5
# -------------------------------------------------------------------------------
source = FFSPSource.from_hdf5('/mnt/krakenschest/home/pxpalacios/100_CompDSSI 2026/01_FFSP/05. FFSP_Model/results_rup_bl_1.h5')

# Use best realization
# source.subfaults = source.best_realization
source.set_active_realization(0)

# -------------------------------------------------------------------------------
# Crustal model (4 layers from FSR)
# -------------------------------------------------------------------------------
# Option 1: Use crust model from FFSP source (commented)
# crustal = source.crust_model

# Option 2: Manual 4-layer model
crustal = CrustModel(4)
vp, vs, rho, thick, Qa, Qb = 1.32, 0.75, 2.40, 0.200, 1000.0, 1000.0
crustal.add_layer(thick, vp, vs, rho, Qa, Qb)
vp, vs, rho, thick, Qa, Qb = 2.75, 1.57, 2.50, 0.800, 1000.0, 1000.0
crustal.add_layer(thick, vp, vs, rho, Qa, Qb)
vp, vs, rho, thick, Qa, Qb = 5.50, 3.14, 2.50, 14.50, 1000.0, 1000.0
crustal.add_layer(thick, vp, vs, rho, Qa, Qb)
vp, vs, rho, thick, Qa, Qb = 7.00, 4.00, 2.67, 0.000, 1000.0, 1000.0
crustal.add_layer(thick, vp, vs, rho, Qa, Qb)

# -------------------------------------------------------------------------------
# Material properties for seismic moment calculation
# -------------------------------------------------------------------------------
ρ = 2500.           # kg / m**3
Vs = 3140.0         # m / s
μ = ρ * Vs **2      # N/m**2

# -------------------------------------------------------------------------------
# Subfault data extraction
# -------------------------------------------------------------------------------
subfaults = source.get_subfaults()
nsubfaults = source.subfaults['npts']

slip_mean = source.all_realizations['slip'].mean()
Area_fault = source.area * 1000 * 1000

# Seismic moment using mean slip
conversion_N_m_to_dyne_cm = 1e7
M0_mean = μ * Area_fault * slip_mean * nsubfaults
Mw_mean = 2/3 * np.log10(M0_mean * conversion_N_m_to_dyne_cm) - 10.7

if rank == 0:
    print(f"Total subfaults: {nsubfaults}")
    print(f"Subfault area: {Area_fault:.2f} m²")
    print(f"Average slip: {slip_mean:.2f} m")
    print(f"M0_mean: {M0_mean:.2e} N·m")
    print(f"Mw_mean: {Mw_mean:.2f}")

# -------------------------------------------------------------------------------
# Station coordinates (UTM)
# -------------------------------------------------------------------------------
utmx = np.array([359958.1764612976, 359909.210734884, 352972.9064965788,
                 356785.00778720574, 357388.4436483849, 343765.088895043,
                 349518.5389304694, 346324.7094952749, 333266.0855400809,
                 337477.02458516695, 336224.1507329495])

utmy = np.array([6294124.525366314, 6302625.576311215, 6302517.54834607,
                 6293263.310727202, 6283866.121857278, 6306996.823725268,
                 6293815.479730421, 6282778.22420124, 6304244.590630547,
                 6292761.800243624, 6278203.501919601])

utm_order = ['Centro', 'H1_s0', 'N1_s1', 'N2_s2', 'N3_s3', 'I1_s4',
             'I2_s5', 'I3_s6', 'F1_s7', 'F2_s8', 'F3_s9']


# -------------------------------------------------------------------------------
# Fault geometry and reference point
# -------------------------------------------------------------------------------
depth_min = source.subfaults['z'].min() / 1e3
fault_dip = source.params['dip']
fault_strike = source.params['strike']
fault_rake = source.params['rake']

x0 = utmx[0] / 1e3
y0 = utmy[0] / 1e3
x_centro_falla = x0 - depth_min / np.tan(fault_dip * np.pi / 180) - source.params['fault_width'] / 2 * np.cos(fault_dip * np.pi / 180)
y_centro_falla = y0 - source.params['fault_width'] / 2 * np.sin(fault_strike * np.pi / 180)
dx0 = x0 - x_centro_falla
dy0 = y0 - y_centro_falla
x0 += dx0
y0 += dy0

# -------------------------------------------------------------------------------
# Shakermaker configuration (FSR parameters)
# -------------------------------------------------------------------------------
dt = 0.005         # Paso de tiempo
nfft = 8192*1       # Numero de muestras del registro
dk = 0.2#.02        # (discretizacion en espacio de longitud de onda) ajustar usando la teoria 
tb = 0              # Cuanto tiempo "adelantar" la ventana de simulacion... no adelantar
tmin = 0.           # tiempo en que comienzan los resultados finales
tmax = 100.         # tiempo en que terminan los resultados finales

# -------------------------------------------------------------------------------
# Build fault source from subfaults
# -------------------------------------------------------------------------------
if rank == 0:
    print("Extracting subfaults...")

x = subfaults['x']
y = subfaults['y']
z = subfaults['z']
slip = subfaults['slip']
strike = subfaults['strike']
dip = subfaults['dip']
rake = subfaults['rake']
rupture_time = subfaults['rupture_time']
rise_time = subfaults['rise_time']

pt_rt = 0.15
MINSLIP = 0
fault_list = []
M0 = 0.0

for i in range(nsubfaults):
    xsrc = x[i] / 1e3 + y0
    ysrc = y[i] / 1e3 + x0
    zsrc = z[i] / 1e3
    phi, theta, lam = strike[i], dip[i], rake[i]
    t0 = rupture_time[i]
    Tr = rise_time[i]
    Tp = pt_rt * Tr
    Te = 0.7 * Tr

    # Accumulate seismic moment (FSR method)
    M0 += μ * Area_fault * slip[i]

    stf = SRF2(Tr=Tr, Tp=Tp, Te=Te, dt=dt, slip=slip[i], a=1.0, b=100.0)
    point_source = PointSource([xsrc, ysrc, zsrc], [phi, theta, lam], tt=t0, stf=stf)
    fault_list.append(point_source)

# Magnitude from accumulated M0
Mw = 2/3 * np.log10(M0 * conversion_N_m_to_dyne_cm) - 10.7

if rank == 0:
    print(f"Created {len(fault_list)} point sources")
    print(f"M0: {M0:.2e} N·m")
    print(f"Mw: {Mw:.2f}")

fault = FaultSource(fault_list, metadata={"name": f"FFSP_Mw{Mw:.2f}"})

# -------------------------------------------------------------------------------
# DRM box parameters
# -------------------------------------------------------------------------------
Lx = 30.0         #km
Ly = 30.0          #km
Lz = 0.030          #km
dx_drm = 500/1000
nx, ny, nz = int(Lx / dx_drm), int(Ly / dx_drm), int(Lz / dx_drm)

# -------------------------------------------------------------------------------
# Green's function parameters
# -------------------------------------------------------------------------------
_m = 0.001
delta_h = 40 * _m
delta_v_rec = 5.0 * _m
delta_v_src = 200 * _m
npairs_max = 200000

# -------------------------------------------------------------------------------
# Run DRM simulation for selected stations
# -------------------------------------------------------------------------------
selected_stations = ['I2_s5']
# 'N2_s2','I2_s5','F2_s8'
for name in selected_stations:
    i = utm_order.index(name)
    drmbox_x0 = [utmy[i] / 1e3, utmx[i] / 1e3, 0]

    drmreceiver = DRMBox(drmbox_x0,
                         [nx, ny, nz],
                         [dx_drm, dx_drm, dx_drm],
                         metadata={"name": name})

    # Disable saving Green's functions for each station
    for j in range(drmreceiver.nstations):
        station = drmreceiver.get_station_by_id(j)
        station.metadata['save_gf'] = False

    writer = DRMHDF5StationListWriter(f"./surface_{name}_sta_{i}_mw6_7.h5drm")
    # gf_databasename = f"./greensfunctions_database_station_{dx_drm*1000}m_{name}_motions_sta_{i}.h5"
    gf_databasename = './greensfunctions_database_surface.h5'


    if rank == 0:
        print(f"DRM Box: {name} at x0={drmbox_x0}, size=({nx}, {ny}, {nz}), dx={dx_drm} km")
        print(f"Database: {gf_databasename}")

    model = shakermaker.ShakerMaker(crustal, fault, drmreceiver)
    model.export_drm_geometry(f"surface_{dx_drm*1000:.0f}m_{name}_geometry_mw6_5.h5drm")

    # Migrate legacy GF database to OP format (adds pair_to_slot, nstations, nsources)
    # Only needs to be done once; safe to call multiple times (overwrites if already present)
    # model.build_pair_to_slot_from_legacy_h5(gf_databasename)

    # Run Stage 2 using precomputed Green's functions database
    model.run_nearest(
        stage='all',
        # stage=1,
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
        sigma=2,
        # Stage 2 only
        tmin=tmin,
        tmax=tmax,
        writer=writer,
        writer_mode='progressive',
        # General
        verbose=False,
        debugMPI=False,
        showProgress=True
    )

if rank == 0:
    print("Done!")
