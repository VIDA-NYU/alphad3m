#!/bin/sh

set -eu

if [ "$#" != "3" ]; then
    echo "Usage: git-lfs-to-squashfs.sh <size> <filename.squashfs> <repo-url>" >&2
    exit 1
fi

SIZE="$1"
FILENAME="$2"
REPO="$3"

# Make an ext3 image
TEMPLATE_DIR="$(mktemp -d)"
mkdir -p "$TEMPLATE_DIR/data"
chmod 777 "$TEMPLATE_DIR/data"
truncate -s "$SIZE" "$FILENAME.ext3"
mkfs.ext3 -d "$TEMPLATE_DIR" "$FILENAME.ext3"
rm -rf "$TEMPLATE_DIR"

# Run singularity
if singularity exec -B "$FILENAME.ext3:/squashfs-workdir:image-src=/" docker://remram/git-lfs-and-mksquashfs bash -c "git clone \"\$1\" /squashfs-workdir/data/repo && mksquashfs /squashfs-workdir/data/repo \"\$2\" -comp lzo" -- "$REPO" "$FILENAME"; then
    # Remove ext3 image
    rm "$FILENAME.ext3"

    echo "Success, built image $FILENAME" >&2
else
    echo "Failed to build image, left ext3 data behind: $FILENAME.ext3" >&2
    exit 1
fi
