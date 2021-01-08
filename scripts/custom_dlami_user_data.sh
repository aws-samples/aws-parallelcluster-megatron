#!/bin/bash
#
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
#

#!/bin/bash
#This script is based on the DLAMI v36 ami-0f899ff8474ea45a9
yum install htop -y 

sudo rm /usr/local/cuda
ln -s /usr/local/cuda-11.0 /usr/local/cuda
export CUDA_HOME=/usr/local/cuda-11.0/
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:$LD_LIBRARY_PATH
export USR_HOME=/home/ec2-user && mkdir -p $USR_HOME && cd $USR_HOME
export PIP_EXEC=$USR_HOME/anaconda3/envs/pytorch_latest_p37/bin/pip
export PYTHON_EXEC=$USR_HOME/anaconda3/envs/pytorch_latest_p37/bin/python

#packages installation
MEGATRON_DIRECTORY=$USR_HOME/megatron
APEX=$USR_HOME/apex

if [ ! -d "$MEGATRON_DIRECTORY" ]; then
    # control will enter here if $DIRECTORY doesn't exist.
    echo "Megatron repository not found. Installing..."
    git clone https://github.com/NVIDIA/Megatron-LM/ $MEGATRON_DIRECTORY
    chown -R ec2-user:ec2-user $MEGATRON_DIRECTORY
    $PIP_EXEC install pipenv transformers dataclasses pybind11 wikiextractor tensorboard jupyterlab
    $PIP_EXEC install -e $MEGATRON_DIRECTORY -U
fi

if [ ! -d $APEX ]; then
    # need to point to a right cuda version and then install latest pytorch
    echo "Apex directory doesn't exist, installing..."
    git clone https://www.github.com/nvidia/apex $APEX
    chown -R ec2-user:ec2-user $APEX
    $PIP_EXEC install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" $APEX 
fi

chown -R ec2-user:ec2-user $USR_HOME

exit $?
