#!/bin/bash

#SBATCH --account=IscrC_REAL-COP
#SBATCH --partition=m100_usr_prod                #m100_all_serial
#SBATCH --qos=normal
#SBATCH -J mpi4py_test
#SBATCH -o mpi4py.out
#SBATCH -e mpi4py.err
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stefano.polo@studenti.unimi.it

module load anaconda/2020.02
module load autoload lapack/3.9.0--gnu--8.4.0
module load openblas/0.3.9--gnu--8.4.0
source activate reinforcement_env


srun -n $SLURM_NTASKS --mpi=pmi2 python run.py
