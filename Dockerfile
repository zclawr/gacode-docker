FROM --platform=linux/amd64 ubuntu:24.04

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

RUN rm -rf /var/lib/apt/lists/* && \
    mkdir /home/user

WORKDIR /home/user

RUN git clone https://github.com/gafusion/gacode.git

COPY exec.LINUX_DOCKER /home/user/gacode/platform/exec/exec.LINUX_DOCKER
COPY make.inc.LINUX_DOCKER /home/user/gacode/platform/build/make.inc.LINUX_DOCKER

RUN echo "export GACODE_PLATFORM=LINUX_DOCKER" >> /../../etc/environment && \
    echo "export GACODE_ROOT=/home/user/gacode" >> /../../etc/environment && \
    echo "export export OMPI_ALLOW_RUN_AS_ROOT=1" >> /../../etc/environment && \
    echo "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1" >> /../../etc/environment

RUN . /home/user/gacode/shared/bin/gacode_setup && \
    . /etc/environment && \
    cd gacode/cgyro && \
    make && \ 
    cd ../tglf && \
    make && \
    cd ../

ENTRYPOINT ["tail", "-f", "/dev/null"]