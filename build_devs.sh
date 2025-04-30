#!/usr/bin/env bash

set -e

echo "[INFO] Building Docker"
_root=${PWD}
this_user=fhrozen
filename=dev_gpu_no_torch.dockerfile

docker build \
  -f "dev_gpu_312.dockerfile" \
  -t "${this_user}/python:gpu-3.12" \
  --build-arg USERNAME="$(whoami)" \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  "${_root}"
