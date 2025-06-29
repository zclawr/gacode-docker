#!/bin/bash

## The BAL Scheduler should behave as follows:

# 1) Run model training on initial dataset
#   - 1.a) During model training, compile inputs to request for simulation
#   - 1.b) Once model training is complete, upload model checkpoint S3 and remember path
#   - 1.c) Send inputs to BAL scheduler
#
# 2) Format inputs s.t. each input.{tglf/cgyro} has its own subdirectory in a parent directory named batch-xyz
#   - 2.a) Upload batch-xyz and its contents to S3
#
# 3) Run batch simulation job, inputting batch-xyz as s3path
#   - 3.a) Simulate all input files in batch-xyz
#   - 3.b) Upload results back to batch-xyz in S3
# 
# 4) Download most recent model checkpoint and batch-xyz from S3 and train
#   - 4.a) Go to step 1.a

# e.g. (WIP)
# Run TGLF simulation jobs
bash ./run_simulation_job.sh tglf gacode/batch-001/tglf/
# Run CGYRO simulation
# bash ../gacode-docker/run_simulation_job.sh cgyro gacode/batch-001/cgyro/
# Wait until jobs complete to run BAL job
kubectl wait --for=condition=complete job --selector=project=ai-fusion-gacode-simulation --timeout=-1s
# Mock BAL run
echo "Ready for active learning with inputs at the s3 path: gacode/batch-001/"

