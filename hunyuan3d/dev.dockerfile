ARG TAG="fhrozen/python"
FROM ${TAG}:gpu-devel-3.10

ARG TORCH_CUDA_ARCH_LIST='8.0;8.6+PTX'
ARG APP_PROFILE=low

ENV FLASH_ATTENTION_SKIP_CUDA_BUILD="TRUE"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV APP_PROFILE=${APP_PROFILE}

WORKDIR /opt
COPY req.txt /opt
RUN pip install -r req.txt && \
    pip install flash-attn --no-build-isolation && \
    rm req.txt && \
    rm -rf /root/.cache/pip && \
    mkdir -p /opt/app

COPY setup_packages.sh ./
RUN --mount=type=bind,rw,source=Hunyuan3D-2GP,target=/opt/app \
    bash setup_packages.sh low
RUN --mount=type=bind,rw,source=Hunyuan3D-2,target=/opt/app \
    bash setup_packages.sh high

WORKDIR /opt/app
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 8080
ENTRYPOINT [ "python", "gradio_app.py"]
CMD [ "--enable_t23d" ]
