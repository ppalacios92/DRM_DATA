#!/bin/bash
#SBATCH --job-name=big_drm
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=16
#SBATCH --mem=0
#SBATCH --output=log_big_drm.txt
pwd; hostname; date
SECONDS=0
source ~/v_ENV/diana_prince//bin/activate

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/mnt/nfshare/lib:$PYTHONPATH
export PYTHONPATH=/mnt/deadmanschest/pxpalacios/v_ENV/diana_prince/lib/python3.10/site-packages:$PYTHONPATH

/opt/openmpi/bin/mpirun python -s 01_surface.py

echo "Elapsed: $SECONDS seconds."
date
