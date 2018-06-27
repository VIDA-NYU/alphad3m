FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-v2018.4.18-20180606-102540
# BUILT FROM https://gitlab.datadrivendiscovery.org/jpl/docker_images/blob/ffc23588281eaa807c167cb4fef6ee906632f183/complete/ubuntu-artful-python36-v2018.4.18.dockerfile
# BUILT FROM https://gitlab.com/datadrivendiscovery/images/blob/1765bc75c730233f10502aa748dfd7df8404fcb7/core/ubuntu-artful-python36-v2018.4.18.dockerfile
# BUILT FROM https://gitlab.com/datadrivendiscovery/images/blob/1765bc75c730233f10502aa748dfd7df8404fcb7/base/ubuntu-artful-python36.dockerfile
# BUILT FROM ubuntu:artful

MAINTAINER "remirampin@gmail.com"

RUN apt-get update -yy && \
    apt-get install -yy git swig && \
    apt-get clean

WORKDIR /usr/src/app
RUN cd /usr/local/src && git clone -n https://gitlab.com/datadrivendiscovery/d3m.git && cd d3m && git checkout 93fe530741c1aa66a8c88b21048b3b413f23ebcc && sed -i 's/2018\.4\.19rc0/2018.4.18/g' d3m/__init__.py setup.py && sed -i 's/install_requires=\[/install_requires=\[] and \[/g' setup.py && pip3 install -e ../d3m
RUN pip3 install Cython==0.28.3
COPY requirements.txt /usr/src/app/requirements.txt
RUN pip3 install -r requirements.txt
COPY d3m_ta2_nyu /usr/src/app/d3m_ta2_nyu
COPY setup.py README.rst /usr/src/app/
RUN pip3 install --no-deps -e /usr/src/app

CMD "ta2_serve"

EXPOSE 50051
