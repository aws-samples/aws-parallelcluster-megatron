#!/bin/bash

# Shared data paths from Fsx for Luster mount point /lustre
DATA_PATH=/lustre/data/wiki/my-gpt2_text_document
CHECKPOINT_PATH=/lustre/data/gpt2/checkpoint
VOCAB_FILE=/lustre/data/gpt2/gpt2-vocab.json
MERGES_FILE=/lustre/data/gpt2/gpt2-merges.txt

# Distributed World configuration
MP_SIZE=8
GPUS_PER_NODE=8
DDP_IMPL=torch
MASTER_ADDR=$SLURM_SUBMIT_HOST
MASTER_PORT=6000
NNODES=$SLURM_NTASKS
NODE_RANK=$SLURM_NODEID
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# CUDA, EFA and NCCL configs
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:$LD_LIBRARY_PATH
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_ALGO=ring
export NCCL_DEBUG=INFO
export RDMAV_FORK_SAFE=1

# Distributed args for Pytorch DDP
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# Training:
/home/ec2-user/anaconda3/envs/pytorch_latest_p37/bin/python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       /home/ec2-user/megatron/pretrain_gpt2.py \
       --model-parallel-size $MP_SIZE \
       --DDP-impl $DDP_IMPL \
       --num-layers 42 \
       --hidden-size 4096 \
       --num-attention-heads 32 \
       --batch-size 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 1000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGES_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --distribute-checkpointed-activations \
       --log-interval 50 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --num-workers 2 \
       --fp16 \
       --tensorboard-dir /lustre/logs/gpt2_param8B_nodes16_bs16_sjob${SLURM_JOB_ID}

set +x
