ARG TAG="fhrozen/python"
FROM ${TAG}:gpu-3.10

WORKDIR /opt

ENV CONDA_OVERRIDE_CUDA="12.8"

COPY streamlit_app.py .

RUN git clone https://github.com/VikParuchuri/marker && \
    rm marker/marker/scripts/streamlit_app.py && \
    cp ./streamlit_app.py marker/marker/scripts/ && \
    cd marker && \
    pip install . && \
    pip install streamlit

EXPOSE 8501
WORKDIR /
ENTRYPOINT [ "marker_gui" ]
