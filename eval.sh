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
        exec ta2_search "$D3MINPUTDIR/search_config.json"
    ;;
    test)
        exec "$D3MTESTOPT" "$D3MINPUTDIR/test_config.json"
    ;;
    ta2ta3)
        export CONFIG_JSON="{\"temp_storage_root\": \"$D3MOUTPUTDIR/supporting_files\", \"pipeline_logs_root\": \"$D3MOUTPUTDIR/pipelines\", \"executables_root\": \"$D3MOUTPUTDIR/executables\", \"timeout\": \"$D3MTIMEOUT\", \"cpus\": \"$D3MCPU\", \"ram\": \"$D3MRAM\", \"shared_storage_root\": \"$D3MOUTPUTDIR\"}"
        exec ta2_serve 45042
    ;;
    *)
        echo "\$D3MRUN is not set" >&2
        exit 1
    ;;
esac
