#!/bin/bash
# Parameters: 
# $1 : simulation type [tglf, cgyro]
# $2 : path to input folder
. /etc/environment
. $GACODE_ROOT/shared/bin/gacode_setup
for dir in $2; do
  if [[ -d "$dir" ]]; then
    # echo "Processing directory: $dir"
    run_simulation $1 $dir
  fi
done

# Parameters:
# $1 : simulation type [tglf, cgyro]
# $2 : simulation directory
function run_simulation {
    if [[ $1 == "tglf" ]]; then
        echo "Beginning TGLF at $2"
        tglf -i $2
        tglf -e $2
    elif [[ $1 == "cgyro" ]]; then
        echo "Beginning CGYRO at $2"
        cgyro -i $2
        cgyro -e $2
    else
        echo "Please specify either tglf or cgyro"
    fi
}