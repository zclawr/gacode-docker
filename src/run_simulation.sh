#!/bin/bash
. /etc/environment
. $GACODE_ROOT/shared/bin/gacode_setup
if [[ $1 == "tglf" ]]; then
    echo "Beginning TGLF"
    tglf -i sim
    tglf -e sim
elif [[ $1 == "cgyro" ]]; then
    echo "Beginning CGYRO"
    cgyro -i sim
    cgyro -e sim
else
    echo "Please specify either tglf or cgyro"
fi