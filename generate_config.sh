#!/bin/sh

usage(){
    echo "Usage: ./generate_config.sh /tmp/output_dir ../data/185_baseball" >&2
}

# Directory from which this script was called
# This is the directory config_train.json and config_test.json are written
WD="$(pwd)"
# The directory in which TA2 will write
OUTPUT="$1"
if [ -z "$OUTPUT" ]; then usage; exit 1; fi
OUTPUT="$WD/$OUTPUT"
# The path to the dataset. Can be relative to $WD
DATASET="$(cd "$WD" && cd "$2" && pwd)"
if [ -z "$DATASET" ]; then usage; exit 1; fi
# Get dataset name
DATASET_NAME="$(cd "$DATASET" && basename "$(pwd)")"

echo "Dataset: $DATASET_NAME" >&2
echo "Dataset path: $DATASET" >&2
echo "Output directory: $OUTPUT" >&2

cd "$WD"

cat >config_train.json <<END
{
    "temp_storage_root": "$OUTPUT/storage",
    "pipeline_logs_root": "$OUTPUT/logs",
    "executables_root": "$OUTPUT/executables",
    "dataset_schema": "$DATASET/TRAIN/dataset_TRAIN/datasetDoc.json",
    "problem_schema": "$DATASET/TRAIN/problem_TRAIN/problemDoc.json",
    "training_data_root": "$DATASET/TRAIN/dataset_TRAIN",
    "problem_root": "$DATASET/TRAIN/problem_TRAIN",
    "timeout": "30",
    "cpus": "2",
    "ram": "5Gi"
}
END

cat >config_test.json <<END

{
    "temp_storage_root": "$OUTPUT/storage",
    "executables_root": "$OUTPUT/executables",
    "dataset_schema": "$DATASET/TEST/dataset_TEST/datasetDoc.json",
    "problem_schema": "$DATASET/TEST/problem_TEST/problemDoc.json",
    "test_data_root": "$DATASET/TEST/dataset_TEST",
    "problem_root": "$DATASET/TEST/problem_TEST",
    "results_root": "$OUTPUT/results",
    "timeout": "30",
    "cpus": "2",
    "ram": "5Gi"
}
END
