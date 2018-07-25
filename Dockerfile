FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-v2018.7.10-20180721-224737@sha256:22abbbcbd6e793431d0d577df2e0c015650a6243e675815472e33ba1466bf95a
#     Dockerfile: https://gitlab.datadrivendiscovery.org/jpl/docker_images/blob/2afc06e993e0d460ecddac233a226c9ed097a04a/complete/ubuntu-artful-python36-v2018.7.10.dockerfile
#     Using primitives https://gitlab.datadrivendiscovery.org/jpl/primitives_repo/tree/7ed5e2963cd21511466210b9b10eca468dd0a9cd
# BUILT FROM registry.gitlab.com/datadrivendiscovery/images/core:ubuntu-artful-python36-v2018.7.10@sha256:e1ae7db8c1f9b91e57e367b4c029046b4fd391948cb102ec9e49ab7e73035a70
#     Dockerfile: https://gitlab.com/datadrivendiscovery/images/blob/b891dc97b907fcc2c81e33ddc48ced34708b5524/core/ubuntu-artful-python36-v2018.7.10.dockerfile
# BUILT FROM registry.gitlab.com/datadrivendiscovery/images/base@sha256:52753746a2e6c83e8ba7f5067e530529d3c08d39b038aedd39cea0f931e952d6
#     Dockerfile: https://gitlab.com/datadrivendiscovery/images/blob/1765bc75c730233f10502aa748dfd7df8404fcb7/base/ubuntu-artful-python36.dockerfile

MAINTAINER "remirampin@gmail.com"

RUN apt-get update -yy && \
    apt-get install -yy git swig sqlite3 && \
    apt-get clean

WORKDIR /usr/src/app
#RUN cd /usr/local/src && pip3 install -e git+https://gitlab.com/datadrivendiscovery/d3m.git@00000000#egg=d3m
RUN pip3 install Cython==0.28.3
COPY requirements.txt /usr/src/app/requirements.txt
RUN pip3 install -r requirements.txt
COPY d3m_ta2_nyu /usr/src/app/d3m_ta2_nyu
COPY setup.py README.rst /usr/src/app/
RUN pip3 install --no-deps -e /usr/src/app
COPY eval.sh /usr/local/bin/eval.sh

CMD "/usr/local/bin/eval.sh"

EXPOSE 45042
