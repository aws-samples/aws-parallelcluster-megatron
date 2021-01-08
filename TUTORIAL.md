# Megatron on AWS UltraCluster

This repo contains sample scripts to train the NVIDIA Megatron-LM in AWS using AWS ParallelCluster. 

## TODO:
 - [ ] Add call out to technologies and ultracluster, including picture. 
 - [ ] Include link to Megatron paper
 - [ ] Add index

## Setting up the environment 
#### Local 

To deploy a cluster with AWS ParallelCluster you'll need to install the `aws-parallelcluster` cli. From the root of this project execute the following to create and activate a virtual environment with Parallelcluster installed:

```bash
python -m venv .env
source .env/bin/activate
pip install awscli aws-parallelcluster
pcluster version
```

To execute this sample repo, you will need credentials to access the AWS CLI. Refer to the [getting Started documentation](https://docs.aws.amazon.com/parallelcluster/latest/ug/install.html) for more information on setting up ParallelCluster CLI. 

You'll also need [Hashicorp Packer](https://www.packer.io/downloads.html) for building custom AMIs.

#### Cloud 

This sample repo assumes an existing VPC with private and public subnets. For details on how to provision such infrastructure check out the [this tutorial](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/create-public-private-vpc.html). Private subnets are a requirement for running p4d.24xlarge instances with 4 EFA cards. 

Use the following commands to list VPCs, Subnets and AvailabilityZones:

```bash
aws ec2 describe-subnets --query 'Subnets[].{VPC:VpcId,SUBNET:SubnetId,AZ:AvailabilityZone}'
```

This sample also requires an S3 bucket and a EC2 key pair. You can use the following AWS CLI commands to create new ones:

```bash
# Create a EC2 key pair
aws ec2 create-key-pair --key-name lab-ml-key --query KeyMaterial --output text > ~/.ssh/<Your Key Pair name>
chmod 600 ~/.ssh/<Your Key Pair name>

aws s3 mb s3://<Your Bucket name>
``` 

## Building an Custom AMI

[Build a custom AMI](https://docs.aws.amazon.com/parallelcluster/latest/ug/tutorials_02_ami_customization.html) to avoid long provisioning times associated with [post installation scripts](https://docs.aws.amazon.com/parallelcluster/latest/ug/cluster-definition.html#post-install) for Megatron-LM dependencies. 

The base AMI for customization is an AWS Deep Learning AMI (DLAMI). It already provides the required software to run distributed training of large machine learning models, including NVIDIA drivers and CUDA, EFA plugins and the major deep learning frameworks such as PyTorch and Tensorflow, managed in Conda environments. The Conda package manager can also manage the Megatron-LM dependencies.

To build the custom AMI from the root of the sample project, use the `pcluster createami` command as shown in the script [create_custom_ami.sh](./scripts/create_custom_ami.sh). The command uses the DLAMI v38.0 `ami-01a495658aa5f7930` on `us-west-2` region. Modify it according to your specific region.

The command calls for the script [custom_dlami_user_data.sh](./scripts/custom_dlami_user_data.sh), which installs Megatron-LM and its dependencies, including NVIDIA APEX. The instance used for the build is `-i p4d.24xlarge`, as NVIDIA APEX will be compiled to the host's platform during installation. 

This command specifies an alternative config file to the default one: [`-c $(pwd)/configs/base-config-build-ami.ini`](./configs/base-config-build-ami.ini). This argument is not required if a default configuration file was created during the ParallelCluster installation. Modify the argument values between `<...>` accordingly.   

After the build is complete you get the AMI Id printed on screen:

```bash
Custom AMI ami-xxxxxxxxxxxxx created with name megatron-on-pcluster-aws-parallelcluster-2.10.1-amzn2-hvm-x86_64-202101071208

To use it, add the following variable to the AWS ParallelCluster config file, under the [cluster ...] section
custom_ami = ami-xxxxxxxxxxxxx

```

## Configure and Deploy a Cluster

Use the [configs/multi-queue-config.ini](./configs/multi-queue-config.ini) configuration file to stand up the cluster. Make sure to modify the VPC, FSX and COMPUTE sessions accordingly:

```
[vpc megatron]
vpc_id = <Your VPC Id>
master_subnet_id = <A public subnet Id>
compute_subnet_id = <A private subnet Id>
 
 ... 

[fsx sharedfsx]
shared_dir = /lustre
storage_capacity = 1200
import_path =  s3://<Your Bucket Name>
export_path =  s3://<Your Bucket Name>
deployment_type = SCRATCH_2

 ...

[cluster multi-queue-us-west-2]
key_name = <Your EC2 Key pair name>
base_os = alinux2                   # optional, defaults to alinux2
scheduler = slurm
master_instance_type = c5.4xlarge    # optional, defaults to t2.micro
vpc_settings = megatron
scaling_settings = quick
queue_settings = gpu, cpu
custom_ami = <Your Custom AMI Id>
s3_read_write_resource = arn:aws:s3:::<Your Bucket Name>*
compute_root_volume_size = 256
master_root_volume_size = 128
fsx_settings = sharedfsx
```

Create the cluster with: `pcluster create megatron-on-pcluster -c configs/multi-queue-config.ini`

Once peloyment completes you get the Head Node's public and private ips printed on the screen:

```bash
Creating stack named: parallelcluster-megatron-on-pcluster
Status: parallelcluster-megatron-on-pcluster - CREATE_COMPLETE
MasterPublicIP: yyy.yyy.yy.yyy
ClusterUser: ec2-user
MasterPrivateIP: xxx.xxx.xx.xxx
```

Access the cluster Head node using the cli command `pcluster ssh megatron-on-pcluster -i ~/.ssh/<Your Key Pair name>`


## Preparing the Dataset wth CPU instances

Once in the cluster head node, set-up a data folder in the _/lustre_ directory and download the latest English Wikipedia data dump from Wikimedia:

```bash
export DATA_DIR=/lustre/data
mkdir -p $DATA_DIR && cd $DATA_DIR

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

Once the the data is available, provision a cpu node using slurm: `salloc --nnodes 1 -p cpu`.




