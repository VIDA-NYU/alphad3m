FROM ubuntu:16.04
MAINTAINER "remirampin@gmail.com"

RUN apt-get update -yy && \
    apt-get install -yy git python python-pip python-virtualenv python-numpy python-dateutil python-scipy && \
    apt-get clean

# Required by NIST to build Python in our image, apparently needed for their evaluation process (?)
RUN apt-get install -yy build-essential libncursesw5-dev libreadline6-dev libssl-dev libgdbm-dev libc6-dev libsqlite3-dev tk-dev libbz2-dev zlib1g-dev

WORKDIR /usr/src/app
RUN virtualenv --system-site-packages /usr/src/app/venv && . /usr/src/app/venv/bin/activate
RUN /usr/src/app/venv/bin/pip install backports.ssl-match-hostname certifi docutils futures grpcio \
    ipython~=5.4 scikit-learn==0.19 usagestats==0.7 nose && \
    printf "#!/bin/sh\n\n/usr/src/app/venv/bin/ta2_search \"\$@\"\n" >/usr/local/bin/ta2_search && \
    chmod +x /usr/local/bin/ta2_search && \
    printf "#!/bin/sh\n\n/usr/src/app/venv/bin/ta2_serve \"\$@\"\n" >/usr/local/bin/ta2_serve && \
    chmod +x /usr/local/bin/ta2_serve
RUN /usr/src/app/venv/bin/pip install 'pandas>=0.20.1' 'langdetect>=1.0.7' dsbox-datacleaning==0.1.3 dsbox-dataprofiling==0.1.4
COPY vistrails /usr/src/app/vistrails
RUN /usr/src/app/venv/bin/pip install -e /usr/src/app/vistrails
COPY userpackages /usr/src/app/userpackages
COPY d3m_ta2_vistrails /usr/src/app/d3m_ta2_vistrails
COPY setup.py /usr/src/app/setup.py
RUN /usr/src/app/venv/bin/pip install -e /usr/src/app
COPY pipelines /usr/src/app/pipelines

# Workaround for installation issues
RUN /usr/src/app/venv/bin/pip install decorator==4.1.2

EXPOSE 50051
