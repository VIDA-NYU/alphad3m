FROM ubuntu:17.10
MAINTAINER "remirampin@gmail.com"

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update -yy && \
    apt-get install -yy git python3.6 python3-pip python3-virtualenv && \
    apt-get clean

# Required by NIST to build Python in our image, apparently needed for their evaluation process (?)
#RUN apt-get install -yy build-essential libncursesw5-dev libreadline6-dev libssl-dev libgdbm-dev libc6-dev libsqlite3-dev tk-dev libbz2-dev zlib1g-dev

WORKDIR /usr/src/app
RUN python3 -m virtualenv -p python3.6 --system-site-packages /usr/src/app/venv && . /usr/src/app/venv/bin/activate
RUN /usr/src/app/venv/bin/pip install -U certifi pip pip-tools
COPY requirements.txt /usr/src/app/requirements.txt
RUN /usr/src/app/venv/bin/pip-sync requirements.txt
COPY d3m_ta2_nyu /usr/src/app/d3m_ta2_nyu
COPY setup.py /usr/src/app/setup.py
RUN /usr/src/app/venv/bin/pip install --no-deps -e /usr/src/app
RUN printf "#!/bin/sh\n\n/usr/src/app/venv/bin/ta2_search \"\$@\"\n" >/usr/local/bin/ta2_search && \
    chmod +x /usr/local/bin/ta2_search && \
    printf "#!/bin/sh\n\n/usr/src/app/venv/bin/ta2_serve \"\$@\"\n" >/usr/local/bin/ta2_serve && \
    chmod +x /usr/local/bin/ta2_serve

EXPOSE 50051
