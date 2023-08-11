#!/usr/bin/env bash
shopt -s expand_aliases
HERE=$(dirname "$0")

. "automlbenchmark/frameworks/shared/setup.sh" "$HERE" true
export AR=/usr/bin/ar
PIP install Cython==0.29.24
PIP install scipy==1.7.3
PIP install -r "$HERE/requirements.txt"

PY -c "from alphad3m import __version__; print(__version__)" >> "${HERE}/.installed"
