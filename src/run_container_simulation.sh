#!/bin/bash
# $1 : first param is simulation type [tglf, cgyro]
# $2 : second param is the path to input files
# docker pull zclawr/ai-fusion
# docker create --name gacode-sim-tmp gacode-tglf-cgyro
# docker cp $2 gacode-sim-tmp:/home/user/sim
# docker commit gacode-sim-tmp gacode-sim
# docker run gacode-sim $1
echo "HELLO THERE"