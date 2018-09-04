#!/bin/sh

# Example train: ./docker.sh tpl fast search seed_datasets_current/uu4_SPECT/TRAIN ta2-test:latest
# Example test: ./docker.sh test seed_datasets_current/uu4_SPECT/TEST ta2-test:latest
# Example ta2-ta3: ./docker.sh ta3 seed_datasets_current/uu4_SPECT/TRAIN ta2-test:latest

# Change this if you're not Remi
LOCAL_DATA_ROOT="/home/remram/Documents/programming/d3m/data"
LOCAL_OUTPUT_ROOT="/home/remram/Documents/programming/d3m/tmp"

set -eu

OPTS=""
TIMEOUT=30
if [ "$1" = "tpl" ]; then
    OPTS="$OPTS -e TA2_USE_TEMPLATES=1"
    shift
fi
if [ "$1" = "fast" ]; then
    OPTS="$OPTS -e TA2_DEBUG_BE_FAST=1"
    TIMEOUT=5
    shift
fi
case "$1" in
    ta3)
        MODE=ta2ta3
        INPUT="$2"
        shift 2
    ;;
    search)
        MODE=search
        INPUT="$2"
        shift 2
    ;;
    test)
        MODE=test
        INPUT="$2"
        set -- "$3" bash -c "cd /output/executables; for i in *; do D3MTESTOPT=/output/executables/\$i eval.sh; done"
    ;;
    *)
        echo "Usage:\n  $(basename $0) ta3 seed_datasets_current/uu4_SPECT/TRAIN <image>" >&2
        echo "  $(basename $0) search seed_datasets_current/uu4_SPECT/TRAIN <image>" >&2
        echo "  $(basename $0) test seed_datasets_current/uu4_SPECT/TEST 0123-4567-89abcdef <image>" >&2
        exit 2
    ;;
esac

docker run -ti --rm \
    -p 45042:45042 \
    -e D3MRUN="$MODE" \
    -e D3MINPUTDIR=/input \
    -e D3MOUTPUTDIR=/output \
    -e D3MCPU=4 \
    -e D3MRAM=4 \
    -e D3MTIMEOUT=$TIMEOUT \
    $OPTS \
    -v "$PWD/d3m_ta2_nyu:/usr/src/app/d3m_ta2_nyu" \
    -v "$LOCAL_DATA_ROOT/${INPUT}:/input" \
    -v "$LOCAL_OUTPUT_ROOT:/output" \
    -v "$PWD/search_config.json:/input/search_config.json" \
    -v "$PWD/test_config.json:/input/test_config.json" \
    "$@"
