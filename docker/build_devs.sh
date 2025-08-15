#!/usr/bin/env bash

set -e

echo "[INFO] Building Docker"
_root=${PWD}
this_user=fhrozen
filename=dev_gpu_312

# docker build \
#   -f "${filename}.dockerfile" \
#   -t "${this_user}/python:gpu-3.12" \
#   --build-arg USERNAME="$(whoami)" \
#   --build-arg USER_UID="$(id -u)" \
#   --build-arg USER_GID="$(id -g)" \
#   "${_root}"

docker build \
  -f "${filename}_root.dockerfile" \
  -t "${this_user}/python-root:gpu-3.12" \
  "${_root}"