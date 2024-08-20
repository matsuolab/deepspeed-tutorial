#!/bin/bash
#$ -l node_f=2
#$ -l h_rt=2:00:00
#$ -j y
#$ -N tutorial
#$ -cwd

# GPU information
nvidia-smi

# Set environmental variables
source scripts/import-env.sh .env

# Node information
NUM_NODE=$NHOSTS
NUM_GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
NUM_GPUS=$(($NUM_NODE * $NUM_GPUS_PER_NODE))

echo "NUM_NODE:${NUM_NODE}"
echo "NUM_GPUS_PER_NODE:${NUM_GPUS_PER_NODE}"
echo "NUM_GPUS:${NUM_GPUS}"

# module load
module load openmpi/5.0.2-gcc

# Activate virtual environment
cd $PATH_TO_WORKING_DIR
source work/bin/activate

# Create hostfile
echo -n "PE_HOSTFILE:"
cat ${PE_HOSTFILE}
export HOSTFILE=hostfile_$JOB_ID
cat ${PE_HOSTFILE} | awk -v num_gpus=$NUM_GPUS_PER_NODE '{print $1, "slots=" num_gpus}' > $HOSTFILE
export MASTER_ADDR=$(cat ${PE_HOSTFILE} | awk '{print $1; exit}')

echo "MASTER_ADDR:${MASTER_ADDR}"
echo "CUDA_HOME_PATH:${CUDA_HOME_PATH}"
echo "LD_LIBRARY_PATH:${LD_LIBRARY_PATH}"

# Run the training script with DeepSpeed
deepspeed --hostfile $HOSTFILE \
    --launcher OpenMPI \
    --no_ssh_check \
    --master_addr=$MASTER_ADDR \
    train.py \
    --config $PATH_TO_CONFIG_FILE \
    --deepspeed \
    --deepspeed_config $PATH_TO_DS_CONFIG