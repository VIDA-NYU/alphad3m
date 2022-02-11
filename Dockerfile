FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python38-master-20220107-233203

MAINTAINER "remi.rampin@nyu.edu, rlopez@nyu.edu, raoni@nyu.edu"

RUN apt-get update -yy && \
    apt-get install -yy git swig sqlite3 && \
    apt-get clean

WORKDIR /usr/src/app

# Install AlphaD3M
COPY alphad3m /usr/src/app/alphad3m
COPY resource /usr/src/app/resource
COPY setup.py README.md /usr/src/app/

# TODO: Improve this installation
RUN pip3 install -e .

# TODO: Install here all non-d3m dependencies:
# 'SQLAlchemy==1.2.16',
# 'datamart-materialize==0.6.1',
# 'datamart-profiler==0.9',
# 'nltk==3.6.7',
# 'numpy==1.18.2',
# 'scipy==1.4.1',
# 'smac==0.13.1',
# 'scikit-learn==0.22.2.post1',
# 'scikit-image==0.17.2',
# 'torch==1.7',
# 'PyYAML==5.1.2',
# 'metalearn',

COPY eval.sh /usr/local/bin/eval.sh

ARG VERSION
ARG GIT_COMMIT

LABEL version=$VERSION \
    commit=$GIT_COMMIT

CMD "/usr/local/bin/eval.sh"

EXPOSE 45042
