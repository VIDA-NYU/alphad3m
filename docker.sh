#!/bin/sh

# Example train: ./docker.sh ta2-test:latest ta2_search /d3m/config_train.json
# Example test: ./docker.sh ta2-test:latest /d3m/out/executables/50f99cdf-97f7-45d7-8095-35778c39093f /d3m/config_test.json

docker run -ti --rm \
    -v /Users/remram/Documents/programming/d3m/data:/d3m/data \
    -v /Users/remram/Documents/programming/d3m/tmp:/d3m/out \
    -v "$PWD/d3m_ta2_nyu:/usr/src/app/d3m_ta2_nyu" \
    -v "$PWD/config_train.json:/d3m/config_train.json" \
    -v "$PWD/config_test.json:/d3m/config_test.json" \
    "$@"
