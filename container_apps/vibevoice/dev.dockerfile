ARG TAG="fhrozen/python"
FROM ${TAG}:gpu-3.12

RUN mkdir /workspace
WORKDIR /workspace

ENV FLASH_ATTENTION_SKIP_CUDA_BUILD="TRUE"

RUN git clone https://github.com/microsoft/VibeVoice && \
    pip install -e ./VibeVoice && \
    pip install flash-attn --no-build-isolation && \
    rm -rf ./VibeVoice/demo && \
    rm -rf /root/.cache/pip 

WORKDIR /workspace/VibeVoice/demo
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860
ENTRYPOINT [ "python", "gradio_demo.py", "--model_path", "microsoft/VibeVoice-1.5B"]
