ARG TAG="fhrozen/python"
FROM ${TAG}:gpu-3.12

WORKDIR /workspaces

ENV CONDA_OVERRIDE_CUDA="12.8"

COPY marker/ ./marker/

WORKDIR /workspaces/marker
RUN pip install -e . && \
    pip install streamlit && \
    rm -rf marker/marker

EXPOSE 8501
WORKDIR /
ENTRYPOINT [ "marker_gui" ]
