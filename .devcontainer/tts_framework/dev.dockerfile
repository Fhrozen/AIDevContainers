ARG TAG="fhrozen/python"
FROM ${TAG}:gpu-3.12

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
        espeak-ng \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

COPY req.txt /opt
ENV CONDA_OVERRIDE_CUDA="12.8"

RUN conda install -y conda-forge::onnxruntime=1.20.1=py310h430d77f_200_cuda  && \
    pip install -r req.txt && \
    rm req.txt && \
    rm -rf /root/.cache/pip && \
    mkdir -p /opt/app

# Detect the Python site-packages version.
RUN PY_SITEPKG_VER="$(python --version | sed -E 's,^[^0-9]*?([0-9]+\.[0-9]+).*$,\1,')" && \
    for lib in cuda_runtime cublas cudnn cufft curand cuda_nvrtc; do \
        LD_LIBRARY_PATH="/workspaces/venv/lib/python${PY_SITEPKG_VER}/site-packages/nvidia/${lib}/lib/:${LD_LIBRARY_PATH}"; \
    done && \
    echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> ${HOME}/.bashrc

WORKDIR /opt/app
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
EXPOSE 8501
ENTRYPOINT [ "streamlit", "run", "app.py" ]
