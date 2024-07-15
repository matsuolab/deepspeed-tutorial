#!/bin/bash
#$ -l rt_F=9
#$ -l h_rt=0:7:00
#$ -l USE_SSH=1
#$ -j y
#$ -N tutorial
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

# Set environmental variables
NUM_NODE=$NHOSTS
NUM_GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
NUM_GPUS=$(($NUM_NODE * $NUM_GPUS_PER_NODE))

export HOSTFILE=hostfile_$JOB_ID
cat ${SGE_JOB_HOSTLIST} | awk -v num_gpus=$NUM_GPUS_PER_NODE '{print $0, "slots=" num_gpus}' > $HOSTFILE
export MASTER_ADDR=$(cat ${SGE_JOB_HOSTLIST} | head -n 1)

# Run the training script
deepspeed --hostfile $HOSTFILE \
    --launcher OpenMPI \
    --no_ssh_check \
    --master_addr=$MASTER_ADDR \
    train.py \
    --config $PATH_TO_CONFIG_FILE \
    --deepspeed \
    --deepspeed_config $PATH_TO_DS_CONFIG