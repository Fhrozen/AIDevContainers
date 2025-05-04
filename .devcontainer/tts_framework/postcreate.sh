#!/bin/bash

pip install -r .devcontainer/tts_framework/req_dev.txt
pip install -e ./optimum_code

# Detect the Python site-packages version.
PY_SITEPKG_VER="$(python --version | sed -E 's,^[^0-9]*?([0-9]+\.[0-9]+).*$,\1,')"

# Generate the extra library paths.
for lib in cuda_runtime cublas cudnn cufft curand cuda_nvrtc; do
    LD_LIBRARY_PATH="/workspaces/venv/lib/python${PY_SITEPKG_VER}/site-packages/nvidia/${lib}/lib/:${LD_LIBRARY_PATH}"
done

echo "" >> ${HOME}/.bashrc
echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> ${HOME}/.bashrc
