FROM --platform=linux/amd64 ubuntu:24.04

RUN apt update && apt install -y make rsync git vim
RUN apt install -y python3.12-venv

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
  libopenblas-dev \
  python3-pip

#Download s5cmd as per recommended installation (see https://github.com/Rose-STL-Lab/Zihao-s-Toolbox)
RUN wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_linux_amd64.deb && dpkg -i s5cmd_2.2.2_linux_amd64.deb && rm s5cmd_2.2.2_linux_amd64.deb

#Clean up any installation caches after apt installs, make directories for simulation runs
RUN rm -rf /var/lib/apt/lists/* && \
    mkdir /home/user && \
    mkdir /home/user/sim

WORKDIR /home/user

# Download and install Miniconda
RUN cd ../ && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm -rf /tmp/miniconda.sh

# Add Conda to the PATH
ENV PATH="/usr/local/bin:${PATH}"

# Accept Conda TOS 
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Update Conda and clean up
RUN conda update -y conda && \
    conda clean --all --yes

WORKDIR /
# Copy over private key, and set permissions
# Warning! Anyone who gets their hands on this image will be able
# to retrieve this private key file from the corresponding image layer
RUN mkdir -p /root/.ssh
ADD /.ssh/id_rsa /root/.ssh/id_rsa
ADD /.ssh/id_rsa.pub /root/.ssh/id_rsa.pub
ADD /.ssh/known_hosts /root/.ssh/known_hosts

# Create known_hosts
RUN ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts

# Remove host checking
RUN echo "Host github.com\n\tStrictHostKeyChecking no\n" >> /root/.ssh/config

# Update permissions
RUN chmod 400 /root/.ssh/id_rsa
RUN chmod 400 /root/.ssh/config
RUN chmod 400 /root/.ssh/known_hosts

WORKDIR /home/user/

#Clone this repo and set up conda env (requires conda install)
RUN git clone --depth=1 --branch main https://github.com/zclawr/gacode-docker.git && \
    cd ./gacode-docker && \
    git pull && \
    git submodule update --init --recursive && \ 
    cd ./src/output_parsing/ && \ 
    bash setup.sh

#Clone gacode in preparation for compiling TGLF and CGYRO simulation binaries
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

# #Compile tglf and cgyro binaries
RUN . /home/user/gacode/shared/bin/gacode_setup && \
    . /etc/environment && \
    cd gacode/cgyro && \
    make && \ 
    cd ../tglf && \
    make && \
    cd ../

WORKDIR /home/user/gacode-docker

# Comment this if you want to test the docker container
ENTRYPOINT ["sleep", "infinity"]

#NOTES: ---------------
#This only compiles TGLF and CGYRO simulation code; I was having issues with netcdf for some of the other
#simulators. I'm not going to spend time compiling other simulators unless they become necessary.