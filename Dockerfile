FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-devel-20210811-003121

MAINTAINER "remi.rampin@nyu.edu, raoni@nyu.edu, rlopez@nyu.edu"

RUN apt-get update -yy && \
    apt-get install -yy git swig sqlite3 && \
    apt-get clean

WORKDIR /usr/src/app

# Install requirements
COPY requirements.txt /usr/src/app/requirements.txt
RUN pip3 freeze | sort >prev_reqs.txt && \
    pip3 install Cython==0.29.16 && \
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
COPY alphad3m /usr/src/app/alphad3m
COPY resource /usr/src/app/resource
COPY setup.py README.md /usr/src/app/
RUN pip3 freeze | sort >prev_reqs.txt && \
    pip3 install -e /usr/src/app && \
    pip3 freeze | sort >new_reqs.txt && \
    comm -23 prev_reqs.txt new_reqs.txt | while read i; do echo "Removed package $i" >&2; exit 1; done && \
    rm prev_reqs.txt new_reqs.txt

#RUN pip3 install d3m-automl-rpc==1.0.0
RUN pip3 install git+https://gitlab.com/datadrivendiscovery/automl-rpc.git@dev-dist-python
RUN pip3 install nltk==3.4.5
RUN pip3 install datamart-profiler==0.9
RUN pip3 install datamart-materialize==0.6.1

COPY eval.sh /usr/local/bin/eval.sh

ARG VERSION
ARG GIT_COMMIT

LABEL version=$VERSION \
    commit=$GIT_COMMIT

CMD "/usr/local/bin/eval.sh"

EXPOSE 45042
