FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install build-essential wget git vim curl

RUN apt-get -y install python3 python-is-python3
RUN apt-get -y install python3-pip

# Only way to successfully install h5py is through conda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
RUN bash Miniconda3-latest-Linux-aarch64.sh -b
RUN root/miniconda3/bin/conda init
RUN rm Miniconda3-latest-Linux-aarch64.sh

# Needed for Tensorflow to work
RUN root/miniconda3/bin/conda install -y h5py

RUN root/miniconda3/bin/conda install geopandas geoplot rasterio

RUN apt-get -y update
RUN apt-get install -y npm
RUN npm install -g aws-cdk
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash


# https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
# chose what works for an M1 Mac
RUN apt-get install -y unzip
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" 
RUN unzip awscliv2.zip
RUN ./aws/install
RUN rm -rf awscliv2.zip aws

RUN apt-get install -y default-jre