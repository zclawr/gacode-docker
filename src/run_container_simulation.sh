#!/bin/bash
docker create --name gacode-sim-tmp gacode-docker
docker cp $2 gacode-sim-tmp:/home/user/sim
docker commit gacode-sim-tmp gacode-sim
docker run gacode-sim $1