FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.4.4-20190507-022543

MAINTAINER "remi.rampin@nyu.edu"

RUN apt-get update -yy && \
    apt-get install -yy git swig sqlite3 && \
    apt-get clean

WORKDIR /usr/src/app

# Install requirements
COPY requirements.txt /usr/src/app/requirements.txt
RUN pip3 freeze | sort >prev_reqs.txt && \
    pip3 install Cython==0.29.7 && \
    pip3 install -r requirements.txt && \
    pip3 freeze | sort >new_reqs.txt && \
    comm -23 prev_reqs.txt new_reqs.txt | while read i; do echo "Removed package $i" >&2; exit 1; done && \
    rm prev_reqs.txt new_reqs.txt

# Install alphaautoml
COPY alphaautoml /usr/src/app/alphaautoml
RUN pip3 freeze | sort >prev_reqs.txt && \
    pip3 install -e /usr/src/app/alphaautoml && \
    pip3 freeze | sort >new_reqs.txt && \
    comm -23 prev_reqs.txt new_reqs.txt | while read i; do echo "Removed package $i" >&2; exit 1; done && \
    rm prev_reqs.txt new_reqs.txt

# Install TA2
COPY d3m_ta2_nyu /usr/src/app/d3m_ta2_nyu
COPY setup.py README.rst /usr/src/app/
RUN pip3 freeze | sort >prev_reqs.txt && \
    pip3 install -e /usr/src/app && \
    pip3 freeze | sort >new_reqs.txt && \
    comm -23 prev_reqs.txt new_reqs.txt | while read i; do echo "Removed package $i" >&2; exit 1; done && \
    rm prev_reqs.txt new_reqs.txt

# Install TA3-TA2 API
RUN pip3 install https://gitlab.com/datadrivendiscovery/ta3ta2-api/-/archive/58f2da314ea5fa5c2d517f9c2463a7aea4f719f3/ta3ta2-api-58f2da314ea5fa5c2d517f9c2463a7aea4f719f3.zip

COPY eval.sh /usr/local/bin/eval.sh

CMD "/usr/local/bin/eval.sh"

EXPOSE 45042
