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

echo "Building $IMAGE" >&2

docker build \
    -t "$IMAGE" \
    --build-arg VERSION=$(git describe) \
    --build-arg GIT_COMMIT=$(git rev-parse HEAD) \
    .

echo "Done" >&2
echo "Push using:" >&2
echo "    docker push $IMAGE" >&2
