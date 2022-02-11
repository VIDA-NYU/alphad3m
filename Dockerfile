FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python38-master-20220107-233203

MAINTAINER "remi.rampin@nyu.edu, rlopez@nyu.edu, raoni@nyu.edu"

RUN apt-get update -yy && \
    apt-get install -yy git swig sqlite3 && \
    apt-get clean

WORKDIR /usr/src/app

# Install requirements
# (making sure to install Cython first, it is a build dependency)
# This checks that no dependency already installed (from the base image) gets
# removed or replaced, which would break primitives
COPY requirements.txt /usr/src/app/requirements.txt
RUN pip3 freeze | sort >prev_reqs.txt && \
    pip3 install $(grep -i Cython requirements.txt) && \
    pip3 install -r requirements.txt && \
    pip3 freeze | sort >new_reqs.txt && \
    comm -23 prev_reqs.txt new_reqs.txt | while read i; do echo "Removed package $i" >&2; exit 1; done && \
    rm prev_reqs.txt new_reqs.txt

# Install AlphaD3M
COPY alphad3m /usr/src/app/alphad3m
COPY setup.py README.md /usr/src/app/
RUN pip3 install --no-deps -e .

COPY eval.sh /usr/local/bin/eval.sh

ARG VERSION
ARG GIT_COMMIT

LABEL version=$VERSION \
    commit=$GIT_COMMIT

CMD "/usr/local/bin/eval.sh"

EXPOSE 45042
