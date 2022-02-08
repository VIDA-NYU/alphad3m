FROM ubuntu:latest

RUN apt-get update -yy && \
    apt-get install -yy git python3-pip && \
    apt-get clean

MAINTAINER "yamunak@gmail.com"

WORKDIR /usr/src/app
RUN cd /usr/local/src && pip3 install -e git+https://github.com/byu-dml/metalearn.git#egg=metalearn

COPY requirements.txt /usr/src/app/requirements.txt
RUN pip3 install -r requirements.txt 

COPY alphaAutoMLEdit /usr/src/app/alphaAutoMLEdit
COPY setup.py README.rst /usr/src/app/
RUN pip3 install -e /usr/src/app 

COPY sklearn_test.sh /usr/local/bin/sklearn_test.sh

CMD "/usr/local/bin/sklearn_test.sh"

