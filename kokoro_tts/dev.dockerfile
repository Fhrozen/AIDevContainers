ARG TAG="fhrozen/python"
FROM ${TAG}:gpu-3.10

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
        espeak-ng \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

COPY req.txt /opt

RUN pip install -r req.txt && \
    rm req.txt && \
    rm -rf /root/.cache/pip && \
    mkdir -p /opt/app

WORKDIR /opt/app
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860
ENTRYPOINT [ "python", "app.py" ]
