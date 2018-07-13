FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-v2018.7.10-20180713-063442@sha256:f57338e731c7032fd6ecec843c52cd2e43e54104f0141ac2db590c8ea6910ca8
#     Dockerfile: https://gitlab.datadrivendiscovery.org/jpl/docker_images/blob/9c07165049007be8bf41e64ecac64d801264a7f9/complete/ubuntu-artful-python36-v2018.7.10.dockerfile
#     Using primitives https://gitlab.datadrivendiscovery.org/jpl/primitives_repo/tree/9dac9409fbe12269c3cdad64ac224c5704a13734
# BUILT FROM registry.gitlab.com/datadrivendiscovery/images/core:ubuntu-artful-python36-v2018.7.10@sha256:55a72cc2956cff6cbd35a399a21ceefeab5d4492d81ff7bb99a5b84ff9c5af31
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
