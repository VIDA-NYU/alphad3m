#!/bin/sh

REV=$(git describe --always --tags)

docker build -t registry.datadrivendiscovery.org/ta2/nyu_ta2:$REV . && \
docker tag registry.datadrivendiscovery.org/ta2/nyu_ta2:$REV registry.datadrivendiscovery.org/ta2/nyu_ta2:latest && \

docker push registry.datadrivendiscovery.org/ta2/nyu_ta2:$REV && \
docker push registry.datadrivendiscovery.org/ta2/nyu_ta2:latest

while [ -n "$1" ]; do
    docker tag registry.datadrivendiscovery.org/ta2/nyu_ta2:$REV $1:$REV
    docker tag registry.datadrivendiscovery.org/ta2/nyu_ta2:latest $1:latest

    docker push $1:$REV
    docker push $1:latest

    shift
done
