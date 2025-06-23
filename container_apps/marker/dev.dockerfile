ARG TAG="fhrozen/python"
FROM ${TAG}:gpu-3.10

WORKDIR /opt

ENV CONDA_OVERRIDE_CUDA="12.8"

COPY marker/ ./marker/

WORKDIR /opt/marker
RUN pip install -e . && \
    pip install streamlit && \
    rm -rf marker/marker

EXPOSE 8501
WORKDIR /
ENTRYPOINT [ "marker_gui" ]
