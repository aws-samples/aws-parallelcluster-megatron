#!/bin/bash 
#
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
#
echo "Creating a new AMI"

pcluster createami -ai ami-01a495658aa5f7930 \
	-os alinux2 \
	-ap megatron-on-pcluster- \
	-c $(pwd)/configs/base-config-build-ami.ini \
	-i p4d.24xlarge \
	--post-install file://$(pwd)/scripts/custom_dlami_user_data.sh \
	-r us-west-2

