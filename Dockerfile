FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-v2018.6.5-20180702-110329@sha256:64887863b5b32d5349f5f35b21b69bc36fd5ffdedcdf4dbd14df7e05912aa02f
#     Dockerfile: https://gitlab.datadrivendiscovery.org/jpl/docker_images/blob/ffc23588281eaa807c167cb4fef6ee906632f183/complete/ubuntu-artful-python36-v2018.6.5.dockerfile
#     Using primitives https://gitlab.datadrivendiscovery.org/jpl/primitives_repo/tree/f4d810e9b91711518c0b4bd32c2c3a60cdf73f73
# BUILT FROM registry.gitlab.com/datadrivendiscovery/images/core:ubuntu-artful-python36-v2018.6.5@sha256:e01de4837c9d914cb462a2a0c0e7b1f81e31bc555f4f942467ed875c1a22a4a5
#     Dockerfile: https://gitlab.com/datadrivendiscovery/images/blob/dd409b2de02cf74695732ad37039723a5681572a/core/ubuntu-artful-python36-v2018.6.5.dockerfile
# BUILT FROM registry.gitlab.com/datadrivendiscovery/images/base@sha256:52753746a2e6c83e8ba7f5067e530529d3c08d39b038aedd39cea0f931e952d6
#     Dockerfile: https://gitlab.com/datadrivendiscovery/images/blob/1765bc75c730233f10502aa748dfd7df8404fcb7/base/ubuntu-artful-python36.dockerfile

MAINTAINER "remirampin@gmail.com"

RUN apt-get update -yy && \
    apt-get install -yy git swig && \
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
