FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-v2018.4.18-20180430-150412
# BUILT FROM https://gitlab.datadrivendiscovery.org/jpl/docker_images/blob/f0ad2ee54ca1270d0b6fa3bf212b8f778e61a616/complete/ubuntu-artful-python36-v2018.4.18.dockerfile
# BUILT FROM https://gitlab.com/datadrivendiscovery/images/blob/136a1cb7c21375349a0f149b702a6fb9dfe681bd/core/ubuntu-artful-python36-v2018.4.18.dockerfile
# BUILT FROM https://gitlab.com/datadrivendiscovery/images/blob/136a1cb7c21375349a0f149b702a6fb9dfe681bd/base/ubuntu-artful-python36.dockerfile
# BUILT FROM ubuntu:artful

MAINTAINER "remirampin@gmail.com"

RUN apt-get update -yy && \
    apt-get install -yy git swig && \
    apt-get clean

WORKDIR /usr/src/app
RUN pip3 install Cython==0.27.3
COPY requirements.txt /usr/src/app/requirements.txt
RUN pip3 install -r requirements.txt
COPY d3m_ta2_nyu /usr/src/app/d3m_ta2_nyu
COPY setup.py README.rst /usr/src/app/
RUN pip3 install --no-deps -e /usr/src/app

CMD "ta2_serve"

EXPOSE 50051
