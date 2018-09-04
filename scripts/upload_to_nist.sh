#!/bin/sh

# Change this if you're not Remi
LOCAL_OUTPUT_ROOT="/home/remram/Documents/programming/d3m/tmp"

set -eu

PROBLEM="$1"
TOKEN="$2"

cd "$LOCAL_OUTPUT_ROOT"
BUILD_DIR="$(mktemp -t -d d3m_ta2_submission.XXXXXXXXXX)"
BUILD_FILE="$BUILD_DIR/$PROBLEM.tgz"
tar zcf "$BUILD_FILE" executables pipelines pipelines_considered predictions supporting_files
curl -X POST \
    --header 'Content-Type: multipart/form-data' \
    --header 'Accept: application/json' \
    --header "Authorization: Bearer $TOKEN" \
    -F team_name='New York University' \
    -F track_name=TA2 \
    -F phase_name=Development \
    -F system_name=nyu-ta2 \
    -F problem_id="$PROBLEM" \
    -F file=@"$BUILD_FILE" \
    https://d3m-dse.nist.gov/emp_api_v1/uploads -k
rm -rf "$BUILD_DIR"
