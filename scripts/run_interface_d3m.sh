#!/bin/sh

docker run  -it \
    -v "/Users/rlopez/D3M/examples/input/model.pkl:/input/model.pkl" \
    -v "/Users/rlopez/D3M/examples/input/new_data:/data" \
    -v "/Users/rlopez/D3M/examples/output:/output" \
    registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2019.11.10-20191127-050901 python3 -m d3m runtime produce --fitted-pipeline /input/model.pkl --test-input /data/datasetDoc.json --output /output/predictions.csv
