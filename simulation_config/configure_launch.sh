#!/bin/bash
cp ./launch_template.yml ./launch.yml
echo "inputs:
  default:
    hparam:
      _s3path: $2
run:
  simulation: [$1]
  inputs: [default]" >> ./launch.yml