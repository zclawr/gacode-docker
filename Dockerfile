#
# Ubuntu Dockerfile
#
# https://github.com/dockerfile/ubuntu
#

# Pull base image.
FROM ubuntu:14.04

# Install.
RUN \
  rm -rf /var/lib/apt/lists/* && \
  sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y build-essential && \
  apt-get install -y software-properties-common && \
  apt-get install -y byobu curl git htop man unzip vim wget && \
  apt-get install -y --no-install-recommends gfortran mpich libmpich-dev libfftw3-dev && \
  apt-get install -y --no-install-recommends python3 python-is-python3 && \
  apt-get install -y --no-install-recommends openmpi-bin openmpi-doc libopenmpi-dev && \
  apt-get install -y --no-install-recommends libopenblas-dev && \
  rm -rf /var/lib/apt/lists/*

# Set environment variables.
ENV HOME /root

# Define working directory.
WORKDIR /root

# Define default command.
CMD ["bash"]