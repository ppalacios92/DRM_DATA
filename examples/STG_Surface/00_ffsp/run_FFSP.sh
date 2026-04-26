#!/bin/bash
#SBATCH --job-name=FFSP_compDDSI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=log_FFSP.txt
pwd; hostname; date
SECONDS=0
source ~/v_ENV/barry_allen/bin/activate

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/mnt/nfshare/lib:$PYTHONPATH
export PYTHONPATH=/mnt/deadmanschest/pxpalacios/REPO/ShakerMaker:$PYTHONPATH

/opt/openmpi/bin/mpirun python -s FFSP_model_RUPBL1.py

echo "Elapsed: $SECONDS seconds."
date
