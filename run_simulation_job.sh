#!/bin/bash
# $1 : simulation type [tglf, cgyro]
# $2 : s3 path to inputs (excluding leading slash, but include trailing slash (if pointing to directory))
cp ./config/launch_template.yaml ./config/launch.yaml
echo "dataset:
  default:
    hparam:
      _s3path: $2
run:
  model: [$1]
  dataset: [default]" >> ./config/launch.yaml
echo "Configured launch.yaml, making job"
make job