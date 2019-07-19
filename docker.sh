#!/bin/sh

# Example search: ./docker.sh fast search seed_datasets_current/185_baseball/TRAIN ta2:latest
# Example ta2ta3: ./docker.sh ta2ta3 seed_datasets_current/185_baseball/ ta2:latest
# Example test: ./docker.sh test seed_datasets_current/185_baseball/TEST ta2:latest

# Change this if you're not Remi
LOCAL_DATA_ROOT="/Users/raonilourenco/reps/datasets"
LOCAL_OUTPUT_ROOT="/Users/raonilourenco/d3m/tmp"
LOCAL_STATIC_ROOT="/Users/raonilourenco/d3m/static"


set -eu

OPTS=""
TIMEOUT=10
while true; do
    if [ "$1" = "fast" ]; then
        OPTS="$OPTS -e TA2_DEBUG_BE_FAST=1"
        TIMEOUT=5
        shift
    else
        break
    fi
done
case "$1" in
    search)
        MODE=search
        INPUT="$2"
        shift 2
    ;;
    ta2ta3)
        MODE=ta2ta3
        INPUT="$2"
        shift 2
    ;;
    test)
        MODE=test
        INPUT="$2"
        set -- "$3" bash -c "cd /output/executables; for i in *; do D3MTESTOPT=/output/executables/\$i eval.sh; done"
    ;;
    *)
        echo "Usage:\n  $(basename $0) search seed_datasets_current/185_baseball/TRAIN <image>" >&2
        echo "  $(basename $0) ta2ta3 seed_datasets_current/185_baseball/TRAIN <image>" >&2
        echo "  $(basename $0) test seed_datasets_current/185_baseball/TEST 0123-4567-89abcdef <image>" >&2
        exit 2
    ;;
esac

docker run -ti --rm \
    -p 45042:45042 \
    -e D3MRUN="$MODE" \
    -e D3MINPUTDIR=/input \
    -e D3MOUTPUTDIR=/output \
    -e D3MSTATICDIR=/static \
    -e D3MCPU=4 \
    -e D3MRAM=4Gi \
    -e D3MTIMEOUT=$TIMEOUT \
    $OPTS \
    -v "$PWD/alphaautoml:/usr/src/app/alphaautoml" \
    -v "$PWD/d3m_ta2_nyu:/usr/src/app/d3m_ta2_nyu" \
    -v "$PWD/resource:/usr/src/app/resource" \
    -v "$PWD/tests.py:/usr/src/app/tests.py"\
    -v "$PWD/eval.sh:/usr/local/bin/eval.sh"\
    -v "$LOCAL_DATA_ROOT/${INPUT}:/input" \
    -v "$LOCAL_OUTPUT_ROOT:/output" \
    -v "$LOCAL_STATIC_ROOT:/static" \
    --name ta2_container \
    "$@"
