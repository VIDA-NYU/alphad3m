#!/bin/sh

# This is the entrypoint used by Data Machines
# It reads the environment variables and calls AlphaD3M correctly


if [ -z "$D3MINPUTDIR" ]; then
    echo "\$D3MINPUTDIR is not set" >&2
    exit 1
fi

case "$D3MRUN"
in
    ta2ta3)
        exec alphad3m_serve
    ;;
    *)
        echo "\$D3MRUN is not set" >&2
        exit 1
    ;;
esac
