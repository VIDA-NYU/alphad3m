#!/bin/sh

set -eu

if [ $# = 1 ]; then
    if echo "$1" | grep -q "[/:]"; then
        # If it contains "/" or ":" it's a full image name
        IMAGE="$1"
    else
        # Otherwise we assume it's just a tag
        IMAGE="registry.gitlab.com/vida-nyu/d3m/alphad3m:$1"
    fi
else
    # Default
    IMAGE="registry.gitlab.com/vida-nyu/d3m/alphad3m:$(git describe)"
fi

docker build -t "$IMAGE" .
