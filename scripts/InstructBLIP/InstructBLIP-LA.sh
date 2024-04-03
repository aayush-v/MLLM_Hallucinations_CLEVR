#!/bin/bash

#SBATCH -G a100:1
#SBATCH --mem=128G
#SBATCH -N 1            # number of nodes
#SBATCH -c 4            # number of cores 
#SBATCH -t 2-00:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required modules for job's environment
module load mamba/latest
# Using python, so source activate an appropriate environment
source activate mllm_hallucination

aug_type="$1"

python InstructBLIP-LA.py "$aug_type"


