from shakermaker.crustmodel import CrustModel
from shakermaker.ffspsource import FFSPSource
import numpy as np

# MPI:: Armar MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

if rank ==0:
    # Model crustal
    crustal = CrustModel(3)
    # thickness, vp, vs, rho, Qa, Qb
    crustal.add_layer(15.5, 5.5, 3.14, 2.5, 1000.0, 1000.0)
    crustal.add_layer(31.5, 7.0, 4.0, 2.67, 1000.0, 1000.0)
    crustal.add_layer(0.0, 8.0, 4.57, 2.8, 1000.0, 1000.0)

    # Create FFSP source with all parameters from your .inp
    source = FFSPSource(
        id_sf_type=8,  freq_min=0.01,  freq_max=24.0,
        fault_length=30.0,   fault_width=16.0,
        x_hypc=15.0,  y_hypc=8.0,  depth_hypc=8.0,
        xref_hypc=0.0,  yref_hypc=0.0,
        magnitude=6.7,  fc_main_1=0.09,  fc_main_2=3.0, rv_avg=3.0,
        ratio_rise=0.3,
        strike=358.0,  dip=40.0,  rake=113.0,
        pdip_max=15.0,   prake_max=30.0,
        nsubx=256,   nsuby=128,
        nb_taper_trbl=[5, 5, 5, 5],
        seeds=[52, 448, 4446],
        id_ran1=1,  id_ran2=16,
        angle_north_to_x=0.0,
        is_moment=3,
        crust_model=crustal,
        verbose=True,
    )
else:
    crustal = None
    source = None

# MPI:: Compartir con todos los ranks la info de crustal y source
crustal = comm.bcast(crustal, root=0)
source = comm.bcast(source, root=0)
# Run FFSP en paralelo
comm.Barrier() 
subfaults = source.run()
comm.Barrier()
# ppp

# Guardar archivos (solo rank 0 los escribe)
if rank ==0:
    source.write_ffsp_format('./output_ffsp_rup_bl_1')
    source.write_hdf5('./results_rup_bl_1.h5')