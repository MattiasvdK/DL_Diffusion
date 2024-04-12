#!/bin/bash
#SBATCH --job-name=DL
#SBATCH --output=DL.out
#SBATCH --time=40:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=12GB
#SBATCH --partition=gpu

mkdir $TMPDIR/dataset
mkdir -p $TMPDIR/results/log $TMPDIR/results/model

# Copy Dataset
unzip /scratch/$USER/train2014.zip -d $TMPDIR/dataset
unzip /scratch/$USER/test2014.zip -d $TMPDIR/dataset
unzip /scratch/$USER/val2014.zip -d $TMPDIR/dataset

# Copy to TPMDIR
cp /home4/$USER/DL_Diffusion $TMPDIR
cd $TMPDIR

# Run the training
/home4/$USER/venvs/rizkienvs/bin/python3 src/main.py

mkdir -p /home4/$USER/QA_jobs/job_${SLURM_JOBID}
tar czvf /home4/$USER/QA_jobs/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results