#!/bin/bash
#
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
#

echo "Deep Learning AMI ID $DEEP_LEARNING_AMI_ID"

echo "Creating a new AMI"

pcluster createami -ai $DEEP_LEARNING_AMI_ID \
	-os alinux2 \
	-ap megatron-on-pcluster- \
	-c $(pwd)/configs/base-config-build-ami.ini \
	-i p4d.24xlarge \
	--post-install file://$(pwd)/scripts/custom_dlami_user_data.sh \
	-r ${AWS_REGION}
