#!/bin/sh

set -eu

# Change this to the correct image
ALPHAD3M_IMAGE="docker://registry.gitlab.com/vida-nyu/d3m/alphad3m:devel"

# Change this to the correct paths
ALPHAD3M_SINGULARITY="$HOME/alphad3m-singularity.sh"
ALPHAD3M="$HOME/alphad3m"

if [ "$#" "<" 1 ]; then
    echo "Usage: run-jobs.sh <dataset1> [dataset2...]" >&2
    exit 1
fi

if [ ! -e "$ALPHAD3M_SINGULARITY" ]; then
    echo "alphad3m-singularity.sh not found" >&2
    exit 1
fi
echo "Using alphad3m-singularty.sh at $ALPHAD3M_SINGULARITY" >&2

if [ ! -e "$ALPHAD3M" ]; then
    echo "alphad3m not found" >&2
    exit 1
fi
echo "Using alphad3m at $ALPHAD3M" >&2

echo "Using image $ALPHAD3M_IMAGE" >&2

for dataset in "$@"; do
    if [ -e "${dataset}" ]; then
        echo "Folder exists: ${dataset}" >&2
        exit 1
    fi
    mkdir "${dataset}"
    cat > "${dataset}/job.sh" <<END
#!/bin/sh

"$ALPHAD3M_SINGULARITY" "$ALPHAD3M_IMAGE" "$dataset"
END
    (cd "${dataset}" && sbatch --time=3:00:00 --mem=30000 --cpus-per-task=4 job.sh)
done
