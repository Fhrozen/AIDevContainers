#!/bin/bash

_PROFILE=$1

if [ "${_PROFILE}" == "${APP_PROFILE}" ]; then
    cd app/hy3dgen/texgen/custom_rasterizer
    pip install .
    cd ../differentiable_renderer
    pip install .
fi
