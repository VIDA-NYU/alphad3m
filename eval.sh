#!/bin/sh

# This is the entrypoint used by Data Machines
# It reads the environment variables and calls the TA2 correctly

# See https://datadrivendiscovery.org/wiki/display/gov/2018+Summer+Evaluation+-+Execution+Process

if [ -z "$D3MINPUTDIR" ]; then
    echo "\$D3MINPUTDIR is not set" >&2
    exit 1
fi

case "$D3MRUN"
in
    search)
        exec ta2_search
    ;;
    ta2)
        exec ta2_serve 45042
    ;;
    ta2ta3)
        exec ta2_serve 45042
    ;;
    test)
        exec "$D3MTESTOPT" "$D3MINPUTDIR/test_config.json"
    ;;
    *)
        echo "\$D3MRUN is not set" >&2
        exit 1
    ;;
esac
