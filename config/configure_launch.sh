#!/bin/bash
# $1 : simulation type [tglf, cgyro]
# $2 : s3 path to inputs (no leading slash)
cp ./launch_template.yml ./launch.yml
echo "dataset:
  default:
    hparam:
      _s3path: $2
run:
  model: [tglf]
  dataset: [default]" >> ./launch.yml