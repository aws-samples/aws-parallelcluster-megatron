# Megatron on AWS UltraCluster

This tutorial walks through the end-to-end process of configuring a cluster with AWS ParallelCluster, with a customized Deep Learning AMI, preprocessing a large dataset using large CPU Amazon EC2 instances and training a GPT-2 Natural Language Understanding model using an AWS EC2 UltraCluster.

Familiarity with AWS cloud concepts of Virtual Private Cloud (VPC), e.g Subnets and Availability Zones, the AWS CLI and bash scripting is recommended.

## Contents:

* [Contents](<#contents>)
* [ParallelCluster Management Setup](<#parallelcluster-management-setup>)
  * [Local Environment](<#local-environement>)
  * [AWS](<#aws>)
* [Building an Custom AMI](<#building-an-custom-ami>)
* [Configure and Deploy a Cluster](<#configure-and-deploy-a-cluster>)
* [Preprocessing the Training Dataset wth CPU Instances](<#preprocessing-the-training-dataset-wth-cpu-instances>)
* [Model Parallel Trainin on a p4d.24xlarge UltraCluster](<#model-parallel-trainin-on-a-p4d24xlarge-ultracluster>)
  * [Monitoring Training with Tensorboard](<#monitoring-training-with-tensorboard>)

## Prerequisites

If you don't have Python3 pip already installed, follow the instructions on the [pip installation](<https://pip.pypa.io/en/latest/installing/>) page before running the commands below.

## ParallelCluster Management Setup

#### Local Environment

To deploy a cluster with AWS ParallelCluster you'll need to install the `aws-parallelcluster` cli.
From the root of this project execute the following to create and activate a virtual environment with Parallelcluster installed:

```bash
python3 -m venv .megatron_env
source .megatron_env/bin/activate
python3 -m pip install awscli aws-parallelcluster==2.10.1
pcluster version

# Set AWS Region
export AWS_REGION="us-west-2"
```

To execute this sample repo, you will need credentials to access the AWS CLI. Refer to the [getting Started documentation](<https://docs.aws.amazon.com/parallelcluster/latest/ug/install.html>) for more information on setting up ParallelCluster CLI.

If you don't have Hashicorp Packer already installed, follow the instructions on the [Hashicorp Packer getting started](<https://learn.hashicorp.com/tutorials/packer/getting-started-install?in=packer/getting-started>) page before running the commands below.
It is required to build custom AMIs.

#### AWS

This sample repo assumes an existing VPC with private and public subnets. For details on how to provision such infrastructure check out the [this tutorial](<https://docs.aws.amazon.com/AmazonECS/latest/developerguide/create-public-private-vpc.html>).
Private subnets are a requirement for running p4d.24xlarge instances with 4 EFA cards.
Please note that the private subnet should have a NAT Gateway setup since AWS ParallelCluster and the Megatron ML lab require internet connection.

Use the following commands to list VPCs, Subnets and Availability Zones:

```bash
aws ec2 describe-subnets --query 'Subnets[].{VPC:VpcId,SUBNET:SubnetId,AZ:AvailabilityZone}' --region ${AWS_REGION}
```

Take note of the Ids to properly configure the cluster environment and set the following environment variables:

```bash
VPC_ID=<value>
PUBLIC_SUBNET_ID=<value>
PRIVATE_SUBNET_ID=<value>
```

This sample also requires an S3 bucket and a EC2 key pair. You can use the following AWS CLI commands to create new ones:

```bash
# Create a EC2 key pair
SSH_KEY_NAME="megatron-lab-key"

aws ec2 create-key-pair --key-name ${SSH_KEY_NAME} \
    --query KeyMaterial \
    --region ${AWS_REGION} \
    --output text > ~/.ssh/${SSH_KEY_NAME}

BUCKET_POSTFIX=$(python3 -S -c "import uuid; print(str(uuid.uuid4().hex)[:10])")aws
BUCKET_NAME="megatron-lab-${BUCKET_POSTFIX}"

aws s3 mb s3://${BUCKET_NAME} --region ${AWS_REGION}
```

## Building an Custom AMI

[Build a custom AMI](<https://docs.aws.amazon.com/parallelcluster/latest/ug/tutorials_02_ami_customization.html>) to avoid long provisioning times associated with using [post installation scripts](<https://docs.aws.amazon.com/parallelcluster/latest/ug/cluster-definition.html#post-install>) for Megatron-LM dependencies.

The base AMI for customization is an AWS Deep Learning AMI (DLAMI).
It already provides the amazon required software to run distributed training of large machine learning models, including NVIDIA drivers and CUDA, EFA plugins and the major deep learning frameworks such as PyTorch and Tensorflow, managed in Conda environments.
The Conda package manager can also manage the Megatron-LM dependencies.

To retrieve the AMI ID of the Deep Learning image v38.0 based on Amazon Linux 2 in the region of deployment, you can use the following command

```bash
# Retrieve Deep Learning AMI ID
export DEEP_LEARNING_AMI_ID=`aws ec2 describe-images --owners amazon \
    --query 'Images[*].{ImageId:ImageId,CreationDate:CreationDate}' \
    --filters "Name=name,Values='Deep Learning AMI (Amazon Linux 2) Version 38.0'" \
    --region ${AWS_REGION} \
    | jq -r 'sort_by(.CreationDate)[-1] | .ImageId'`
```

Before building the customer AMI, you will have to modify the argument values between `<...>` accordingly of the base configuration file for AWS ParallelCluster, located in `./configs/base-config-build-ami.ini`.

The command calls for the script [custom\_dlami\_user\_data.sh](<./scripts/custom_dlami_user_data.sh>), which installs Megatron-LM and its dependencies, including NVIDIA APEX.
The instance used for the build is `-i p4d.24xlarge`, as NVIDIA APEX will be compiled to the host's platform during installation.

The instructions below help to set the variable in the AWS ParallelCluster configuration file that is used to build the customer AMI, i.e. `./configs/base-config-build-ami.ini`.

```bash
git clone https://github.com/pixelb/crudini

# Install dependencies
pip3 install iniparse

# Change the cluster configuration file
python3 crudini/crudini --set ./configs/base-config-build-ami.ini "aws" aws_region_name "${AWS_REGION}"
python3 crudini/crudini --set ./configs/base-config-build-ami.ini "vpc megatron" vpc_id "${VPC_ID}"
python3 crudini/crudini --set ./configs/base-config-build-ami.ini "vpc megatron" master_subnet_id "${PUBLIC_SUBNET_ID}"
python3 crudini/crudini --set ./configs/base-config-build-ami.ini "cluster base-config-build-ami" key_name "${SSH_KEY_NAME}"
```

To build the custom AMI from the root of the sample project, use the `pcluster createami` command as shown in the script [create\_custom\_ami.sh](<./scripts/create_custom_ami.sh>).

```bash
./scripts/create_custom_ami.sh
```

After the build is complete you get the AMI Id printed on screen:

```bash
Custom AMI ami-xxxxxxxxxxxxx created with name megatron-on-pcluster-aws-parallelcluster-2.10.1-amzn2-hvm-x86_64-202101071208

To use it, add the following variable to the AWS ParallelCluster config file, under the [cluster ...] section
custom_ami = ami-xxxxxxxxxxxxx
```

Please set the following variable that will be used to setup the AWS ParallelCluster for running Megatron-ML:

```bash
CUSTOM_AMI=<your value>
```

## Configure and Deploy a Cluster

Use the [configs/multi-queue-config.ini](<./configs/multi-queue-config.ini>) configuration file to stand up the cluster.
You can do that manually by changing the VPC, COMPUTE and FSX sections or use the following commands to change the configuration file using `crudini`:

```bash
# Change the cluster configuration file
python3 crudini/crudini --set ./configs/multi-queue-config.ini "aws" aws_region_name "${AWS_REGION}"
python3 crudini/crudini --set ./configs/multi-queue-config.ini "vpc megatron" vpc_id "${VPC_ID}"
python3 crudini/crudini --set ./configs/multi-queue-config.ini "vpc megatron" master_subnet_id "${PUBLIC_SUBNET_ID}"
python3 crudini/crudini --set ./configs/multi-queue-config.ini "vpc megatron" compute_subnet_id "${PRIVATE_SUBNET_ID}"
python3 crudini/crudini --set ./configs/multi-queue-config.ini "cluster multi-queue-us-west-2" key_name "${SSH_KEY_NAME}"
python3 crudini/crudini --set ./configs/multi-queue-config.ini "cluster multi-queue-us-west-2" s3_read_write_resource "arn:aws:s3:::${BUCKET_NAME}"
python3 crudini/crudini --set ./configs/multi-queue-config.ini "cluster multi-queue-us-west-2" custom_ami "${CUSTOM_AMI}"
python3 crudini/crudini --set ./configs/multi-queue-config.ini "fsx sharedfsx" import_path "s3://${BUCKET_NAME}"
python3 crudini/crudini --set ./configs/multi-queue-config.ini "fsx sharedfsx" export_path "s3://${BUCKET_NAME}"
```

You are now ready to create the cluster with:

```bash
pcluster create megatron-on-pcluster -c configs/multi-queue-config.ini
```

Once deployment completes you get the Head Node's public and private IPs printed on the screen at the end of the cluster creation:

```bash
Creating stack named: parallelcluster-megatron-on-pcluster
Status: parallelcluster-megatron-on-pcluster - CREATE_COMPLETE
MasterPublicIP: yyy.yyy.yy.yyy
ClusterUser: ec2-user
MasterPrivateIP: xxx.xxx.xx.xxx
```

Access the cluster Head Node using the CLI command

```bash
pcluster ssh megatron-on-pcluster -i ~/.ssh/${SSH_KEY_NAME}
```

## Preprocessing the Training Dataset wth CPU instances

Once connected to the cluster head node, set-up a data folder in the _/lustre_ directory and download the latest English Wikipedia data dump from Wikimedia.
This process follows the original [Megatron-LM documentation](<https://github.com/NVIDIA/Megatron-LM#datasets>):

```bash
export WIKI_DIR=/lustre/data/wiki
mkdir -p $WIKI_DIR && cd $WIKI_DIR

wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

Download the vocab and merge table files for the desired model. This example uses the GPT-2 model:

```bash
export DATA_DIR=/lustre/data
export GPT2_DATA=${DATA_DIR}/gpt2

mkdir -p ${GPT2_DATA} && cd ${GPT2_DATA}

wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

mkdir -p ${GPT2_DATA}/checkpoint
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O ${GPT2_DATA}/checkpoint/megatron_lm_345m_v0.0.zip
```

Once the data are available, provision a cpu node using slurm: `salloc --nodes 1 -p cpu`.

All data preprocesing work will proceed from the CPU machine. You can check the provisioning status of your new machine using the `squeue` command. Once the status, _ST_, changes to running, _R_, access the CPU machine terminal through ssh with: `ssh cpu-dy-c5n18xlarge-1`.

Extract the downloaded data using WikiExtractor:

```bash
conda activate pytorch_latest_p37
python -m wikiextractor.WikiExtractor --json /lustre/data/wiki/enwiki-latest-pages-articles.xml.bz2 --output /lustre/data/wiki/text/ -q --processes 70 2>&1 | tee wikiextract.out &
```

Wikiextractor first preprocesses the template of all pages sequentially, followed by a Map/Reduce process for extracting the pages and converting to the loose json format required by Megatron-LM.

Once the extraction completes, we merge the text files with:

```bash
cd /lustre/data/wiki
find /lustre/data/wiki/text/ -name wiki* | parallel -m -j 70 "cat {} >> mergedfile.json"
```

The `mergedfile.json` size on disk is 16GB. With it, create the binary data format for Megatron GPT2:

```bash
python /home/ec2-user/megatron/tools/preprocess_data.py \
    --input /lustre/data/wiki/mergedfile.json \
    --output-prefix my-gpt2 \
    --vocab /lustre/data/gpt2/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file /lustre/data/gpt2/gpt2-merges.txt \
    --append-eod \
    --workers 70
```

Refer to [this solution](<https://github.com/NVIDIA/Megatron-LM/issues/62>) if an `IndexError: list index out of range` occurs.

Once all preprocessing is done we can persist the data from FSx back to S3 using a [Data Repository Task](<https://docs.aws.amazon.com/fsx/latest/LustreGuide/export-data-repo-task.html>) from the terminal used to spin up the cluster.
This guarantees that data gets persisted even if the cluster is termindated.

```bash
# Retrieve the FSx for Lustre file system Id
export FSX_ID=$(aws fsx describe-file-systems --query "FileSystems[?LustreConfiguration.DataRepositoryConfiguration.ExportPath=='s3://<Your Bucket Name>'].FileSystemId" --output text)
# Create data repository task
aws fsx create-data-repository-task \
    --file-system-id $FSX_ID \
    --type EXPORT_TO_REPOSITORY \
    --paths data \
    --report Enabled=true,Scope=FAILED_FILES_ONLY,Format=REPORT_CSV_20191124,Path=s3://<Your Bucket Name>/reports
```

You can exit to the original terminal with the `exit` command 2 times: (1) for exiting the `ssh` session on the CPU node, (2) for the `salloc` slurm allocation.

## Model Parallel Trainin on a p4d.24xlarge UltraCluster

In this section you will train the 8 billion parameters version of Megatron-LM GPT-2 model across 64 GPUs - 8 p4d.24xlarge instances. Log back into the cluster head node using `pcluster ssh ...` if not already on the machine.

Start by creating a training script according to the [original documentation](<https://github.com/NVIDIA/Megatron-LM/blob/main/examples/pretrain_gpt_distributed.sh>).
To train using `slurm` on 8 nodes, modify the distributed world configuration section according to the script [scripts/train\_8B\_gpt2.sh](<./scripts/train_8B_gpt2.sh>).
Make sure to include the CUDA, EFA and NCCL environment variables to enable NCCL to communicate between GPUs through AWS EFA using GPU Remote Direct Memory Access.

To drive the `sbatch` execution of the training script, wrap it on a `job.sh` script, using a shared path across all nodes, such as `/lustre/scripts`. :

```bash
#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH -p gpu
#SBATCH -n 8
#SBATCH -N 8
#SBATCH -o out_%j.out
srun /lustre/scripts/train_8B_gpt2.sh
```

Now you can start training by running `sbatch job.sh` from the head node on the cluster. The output from the run will be recorded on the `.out` file on the current folder.
If your job fails with `slurmstepd: error: execve(): /lustre/scripts/train_8B_gpt2.sh: Permission denied`, change the permissions of your scripts with `chmod +x /lustre/scripts/*.sh`.

Inspecting the NCCL logs in the `.out` file expect to find entries that describe the OFI provide to EFA, such as below:

```bash
gpu-dy-p4d24xlarge-10:33337:33337 [0] NCCL INFO NET/OFI Selected Provider is efa
```

### Monitoring training with Tensorboard

The Megatron-LM framework writes tensorboard logs to the `--tensorboard-dir` specified on the training script.The custom AMI built for the cluster has tensorboard installed on the `pytorch_latest_p37` environment used for training.
Use the to start a tensorboard silently and expose it in a specific port:

```bash
python -m tensorboard.main --port=8080 --logdir /lustre/logs --host 0.0.0.0  2>&1 | tee ~/tensorboard.logs &!
```

Using the following `ssh` tunel configuration when connecting to the head node, you can access tensorboard on `localhost:8080`:

```bash
pcluster ssh megatron-on-pcluster -i ~/.ssh/<Your Key Pair name> -L 8080:localhost:8080
```
