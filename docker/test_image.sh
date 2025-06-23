#!/usr/bin/env bash

name=$(basename ${PWD})

echo "Launching container"

docker run -it --rm --gpus all \
    -v ${PWD}:/workspaces/${name} \
    --name test-image \
    --hostname container fhrozen/python:gpu-3.12
