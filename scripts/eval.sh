#!/bin/bash
#$ -l rt_G.small=1
#$ -l h_rt=0:10:00
#$ -j y
#$ -N tutorial_eval
#$ -cwd

# module load
source scripts/import-env.sh .env
source /etc/profile.d/modules.sh
module load python/3.11/3.11.9 
module load cuda/11.7/11.7.1
module load cudnn/8.9/8.9.7 
module load hpcx-mt/2.12

# Activate virtual environment
cd $PATH_TO_WORKING_DIR
source work/bin/activate

python3 eval.py --config $PATH_TO_CONFIG_FILE
