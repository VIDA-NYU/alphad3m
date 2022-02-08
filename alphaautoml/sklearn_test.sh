#!/bin/sh

# This is the entrypoint used by Data Machines
# It reads the environment variables and calls the TA2 correctly

# See https://datadrivendiscovery.org/wiki/display/gov/2018+Summer+Evaluation+-+Execution+Process

export PYTHONPATH=/usr/local/src/metalearn
exec > /output/output.log
exec sklearn_example 
