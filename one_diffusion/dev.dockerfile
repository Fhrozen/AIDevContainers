ARG TAG="fhrozen/python"
FROM ${TAG}:gpu-3.10

WORKDIR /opt

COPY req.txt /opt
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD="TRUE"

RUN pip install -r req.txt && \
    pip install flash-attn --no-build-isolation && \
    rm req.txt && \
    rm -rf /root/.cache/pip

RUN git clone https://github.com/lehduong/OneDiffusion app
COPY app.py /opt/app/

WORKDIR /opt/app
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860
ENTRYPOINT [ "python", "app.py" ]
