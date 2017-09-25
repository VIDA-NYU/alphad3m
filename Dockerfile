FROM ubuntu:16.04
MAINTAINER "remirampin@gmail.com"

WORKDIR /usr/src/app
RUN apt-get update -yy && \
    apt-get install -yy git python python-pip python-virtualenv python-numpy python-dateutil python-scipy && \
    apt-get clean
RUN virtualenv /usr/src/app/venv && . /usr/src/app/venv/bin/activate
RUN pip install backports.ssl-match-hostname certifi docutils futures grpcio ipython~=5.4 scikit-learn==0.19 usagestats==0.7 && \
    printf "#!/bin/sh\n\n/usr/src/app/venv/bin/ta2_search \"\$@\"" >/usr/local/bin/ta2_search && \
    chmod +x /usr/local/bin/ta2_search && \
    printf "#!/bin/sh\n\n/usr/src/app/venv/bin/ta2_serve \"\$@\"" >/usr/local/bin/ta2_serve && \
    chmod +x /usr/local/bin/ta2_serve
COPY vistrails /usr/src/app/vistrails
RUN pip install -e /usr/src/app/vistrails
COPY d3m_ta2_vistrails /usr/src/app/d3m_ta2_vistrails
COPY setup.py /usr/src/app/setup.py
RUN pip install -e /usr/src/app
COPY pipelines /usr/src/app/pipelines

EXPOSE 50051
