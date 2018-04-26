FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-xenial-python36-v2018.1.26-20180418-192927
# BUILT FROM https://gitlab.com/datadrivendiscovery/images/blob/5d36fec8f6de999d569e65cf7dbfbba79fcf2fba/core/ubuntu-xenial-python36-v2018.1.26.dockerfile
# BUILT FROM https://gitlab.com/datadrivendiscovery/images/blob/1615a60aa1875ccf06a5cf90cffa4e59d5a7be45/base/ubuntu-xenial-python36.dockerfile
# BUILT FROM ubuntu:xenial

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
