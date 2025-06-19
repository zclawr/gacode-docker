FROM --platform=linux/amd64 ubuntu:24.04

#Download necessary libraries
RUN apt-get update && apt-get -y upgrade && apt-get install -y \
  build-essential \
  byobu \
  curl \
  git \
  htop \
  man \
  unzip \
  vim \
  wget \
  openssh-client \
  gfortran \
  mpich \
  libmpich-dev \
  libfftw3-dev \
  python3 \
  python-is-python3 \
  openmpi-bin \
  openmpi-doc \
  libopenmpi-dev \
  libopenblas-dev

#Download s5cmd as per recommended installation (see https://github.com/Rose-STL-Lab/Zihao-s-Toolbox)
RUN wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_linux_amd64.deb && dpkg -i s5cmd_2.2.2_linux_amd64.deb && rm s5cmd_2.2.2_linux_amd64.deb

#Clean up any installation caches after apt installs, make directories for simulation runs
RUN rm -rf /var/lib/apt/lists/* && \
    mkdir /home/user && \
    mkdir /home/user/sim

WORKDIR /home/user

RUN git clone https://github.com/gafusion/gacode.git

#Copy platform files and simulation run scripts into image
COPY src/platform/exec.LINUX_DOCKER /home/user/gacode/platform/exec/exec.LINUX_DOCKER
COPY src/platform/make.inc.LINUX_DOCKER /home/user/gacode/platform/build/make.inc.LINUX_DOCKER
COPY src/run_simulation.sh /home/user/run_simulation.sh

#Set environment variables and paths for tglf and cgyro compilation
RUN echo "export GACODE_PLATFORM=LINUX_DOCKER" >> /../../etc/environment && \
    echo "export GACODE_ROOT=/home/user/gacode" >> /../../etc/environment && \
    echo "export export OMPI_ALLOW_RUN_AS_ROOT=1" >> /../../etc/environment && \
    echo "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1" >> /../../etc/environment && \
    chmod +x /home/user/run_simulation.sh

#Compile tglf and cgyro binaries
RUN . /home/user/gacode/shared/bin/gacode_setup && \
    . /etc/environment && \
    cd gacode/cgyro && \
    make && \ 
    cd ../tglf && \
    make && \
    cd ../

#Run simulation script
ENTRYPOINT ["./run_simulation.sh"] 

#NOTES: ---------------
#This only compiles TGLF and CGYRO simulation code; I was having issues with netcdf for some of the other
#simulators. I'm not going to spend time compiling other simulators unless they become necessary.