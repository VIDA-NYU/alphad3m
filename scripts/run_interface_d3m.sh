#!/bin/sh


docker run  -it \
    -v "/Users/rlopez/D3M/datasets/seed_datasets_current:/input" \
    -v "/Users/rlopez/D3M/tmp/:/output" \
    registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.6.7-20190622-073225 python3 -m d3m runtime produce --fitted-pipeline /input/model.pkl --test-input /input/185_baseball/TEST/dataset_TEST/datasetDoc.json --output /output/new_predictions.csv

