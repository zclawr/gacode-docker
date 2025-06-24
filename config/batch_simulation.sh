#!/bin/bash
# $1 : simulation type [tglf, cgyro]
# $2 : s3 path to inputs (no leading slash)
bash ./configure_launch.sh $1 $2
make job