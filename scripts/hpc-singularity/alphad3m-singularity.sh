#!/bin/sh

# Syntax: ./alphad3m-singularity.sh <image> <dataset>

# Pick a port
D3MPORT=$(python3 -c 'import random; print(random.randint(33000, 64000))')
echo "D3MPORT=${D3MPORT}"

IMAGE="$1"
shift

D3MOUTPUTDIR="$(pwd)/output"

cd "$HOME"

singularity exec \
    -B /scratch/rr2369/d3m-datasets.squashfs:/d3m-datasets:image-src=/ \
    -B /scratch/rr2369/d3m-static-files.squashfs:/d3m-static-files:image-src=/ \
    --env D3MPORT=${D3MPORT} \
    --env D3MRUN=ta2ta3 \
    --env D3MINPUTDIR=/d3m-datasets/seed_datasets_current \
    --env D3MOUTPUTDIR="${D3MOUTPUTDIR}" \
    --env D3MSTATICDIR=/d3m-static-files \
    --env D3MCPU=4 \
    --env D3MRAM=4Gi \
    --env D3MTIMEOUT=60 \
    "${IMAGE}" \
    bash -c "set -m; eval.sh & sleep 30; cd alphad3m && python3 tests/test_datasets.py \"\$@\"" -- "$@"
