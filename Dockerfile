FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install build-essential wget git vim curl

RUN apt-get -y install python3 python-is-python3
RUN apt-get -y install python3-pip

RUN pip install jupyterlab==4.0.7 \
                pandas==2.1.2 \
                plotly==5.18.0 \
                tqdm==4.66.1 \
                scipy==1.11.3 \
                scikit-learn==1.3.2 \
                sphinx==7.2.6 \
                sphinx-rtd-theme==2.0.0 \
                pyan3==1.2.0 \
                graphviz==0.20.1

RUN apt-get -y install graphviz graphviz-dev
RUN apt-get -y install pandoc texlive-xetex texlive-fonts-recommended texlive-plain-generic

RUN apt-get -y install locales-all
RUN apt-get -y install libgdal-dev

RUN apt-get -y install apt-transport-https ca-certificates gnupg curl
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN apt-get update && apt-get -y install google-cloud-cli
