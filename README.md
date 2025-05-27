# gacode-docker

In order to run regression tests with the GACode simulators, run the following:

## Building and running the Docker image
Run the following in the root directory of this repository:
```
docker build -t gacode-docker .
docker run gacode-docker
```
This will cause the container to run with an open terminal, which will not close until the container is stopped.

## Running regressions tests
In order to run the regression tests, execute the following commands in the docker container. I recommend navigating to the Docker container you ran in the command above through Docker Desktop, and going to the Exec tab on that container. From there, you can run these commands in the terminal. If you would prefer to run this from a CLI without Docker Desktop involved, you can use ```docker exec ...``` (see [Docker documentation](https://docs.docker.com/reference/cli/docker/container/exec/) for more).

```
. /etc/environment
. $GACODE_ROOT/shared/bin/gacode_setup
tglf -r
cgyro -r -n 4 -nomp 2
```

The first two commands are necessary to propagate environment variables into the terminal.

## Troubleshooting
If you run into the error: ```/bin/sh: 1: tglf: not found```, it is likely that the terminal you are executing commands from does not have necessary environment variables available to it. If you have indeed run the above commands for running regression tests, try running the following in your terminal to manually add the necessary environment variables:
```
export GACODE_PLATFORM=LINUX_DOCKER
export GACODE_ROOT=/home/user/gacode
export export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
. $GACODE_ROOT/shared/bin/gacode_setup
```
From here, try running the regression tests again.