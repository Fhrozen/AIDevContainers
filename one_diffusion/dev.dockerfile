ARG TAG
FROM ${TAG}:gpu-3.10

COPY req.txt /
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD="TRUE"

RUN pip install -r req.txt && \
    pip install flash-attn --no-build-isolation && \
    rm req.txt && \
    rm -rf /root/.cache/pip

RUN mkdir -p /opt/app
