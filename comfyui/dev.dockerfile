ARG TAG="fhrozen/python"
FROM ${TAG}:gpu-3.10

WORKDIR /opt

COPY req.txt /opt
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD="TRUE"

RUN pip install -r req.txt && \
    pip install flash-attn --no-build-isolation && \
    rm req.txt && \
    rm -rf /root/.cache/pip && \
    mkdir /opt/app

WORKDIR /opt/app

EXPOSE 8188
ENTRYPOINT [ "python", "main.py" ]
CMD ["--listen", "0.0.0.0"]
