#!/bin/bash
#SBATCH --job-name=DL
#SBATCH --output=DL.out
#SBATCH --time=40:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=12GB
#SBATCH --partition=gpu

mkdir -p $TMPDIR/results_DL/model_DL_train $TMPDIR/results_DL/model_DL_test 

# Copy to TPMDIR
cp /home4/$USER/DA_model/QA_DL_model.py $TMPDIR/results_DL
cd $TMPDIR/results_DL

# Run the training
/home4/$USER/venvs/rizkienv/bin/python3 QA_DL_model.py 

mkdir -p /home4/$USER/QA_jobs/job_${SLURM_JOBID}
tar czvf /home4/$USER/QA_jobs/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results_DL