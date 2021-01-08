# Megatron on AWS UltraCluster

This is a repo with code for training NVIDIA MEGATRON on AWS P4 UltraCluster.

## Cluster Setup

We use ParallelCluster to provision a multi-queue cluster with large CPU instances (c5m.8xlarge) and large GPU instances (currently p3dn.24xlarge, to be replaced with p4d.24xlarge).

This documentation follows the patterns of creating a cluster as depicted in the [hpcworkshops.com](hpcworkshops.com). Prior to starting, you'll need to create a VPC with dedicated subnets on regions with available p3dn.24xlarge and p4d.24xlarge. Currently IAD1 and IAD7 are recommended. Create also a pem key to access your cluster. 

### Setting up S3

You'll require an S3 bucket to manage data and installations scripts. The 


ls | parallel -m -j $f "cat {} >> ../transactions_cat/transactions.csv"

python3 /shared/megatron/tools/preprocess_data.py \
       --input mergedfile.json \
       --output-prefix my-gpt2 \
       --vocab /lustre/data/gpt2/gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file /lustre/data/gpt2/gpt2-merges.txt \
       --append-eod 
       --workers 60
