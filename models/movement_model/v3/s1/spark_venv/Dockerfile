# DO NOT CHANGE THIS
FROM --platform=linux/amd64 amazonlinux:2023 AS base
# the above must have the same version of python 
# as the EMR version you are using for the following
# to work otherwise it just ignores your packages

RUN yum install -y python3

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3 -m pip install --upgrade pip

# CHANGE THIS TO YOUR REQUIREMENTS
RUN yum install -y git
RUN echo 13

RUN python3 -m pip install \
    venv-pack==0.2.0 \
    git+https://github.com/networkearth/haven@main \
    h3==3.7.7 \
    geopy==2.4.1 \ 
    tensorflow==2.16.1

# DO NOT CHANGE THIS
RUN mkdir /output && venv-pack -o /output/pyspark_ge.tar.gz

FROM scratch AS export
COPY --from=base /output/pyspark_ge.tar.gz /